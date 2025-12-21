import math
import numpy as np

from utils import pixel_to_pitch


class PlayerBallAssigner:
    # Assign ball ownership by nearest-foot distance in meters with configurable stability thresholds
    def __init__(self):
        # Limit maximum ownership radius to avoid assigning the ball to distant players
        self.max_owner_distance_m = 1.4
        # Require a minimum distance gap vs. the second-best candidate to avoid ambiguous ownership
        self.min_margin_distance_m = 0.5

    def _foot_point_meters(self, bbox):
        # Use the bbox bottom-center as a robust proxy for ground contact in camera space
        x1, y1, x2, y2 = bbox
        foot_x = (x1 + x2) / 2
        foot_y = y2
        return pixel_to_pitch(foot_x, foot_y)

    def _ball_center_meters(self, bbox):
        # Use bbox center for ball position since the ball is not anchored to the ground plane point
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        return pixel_to_pitch(cx, cy)

    # Auto-tune ownership thresholds from early frames using robust statistics on nearest-player distances
    def auto_calibrate_from_tracks(self, tracks, max_frames: int = 2000):
        # Collect per-frame nearest-player distances to learn a data-driven ownership radius
        distances = []

        # Limit calibration scope to early frames to keep runtime bounded
        num_frames = min(max_frames, len(tracks["ball"]))

        for frame_idx in range(num_frames):
            ball_dict = tracks["ball"][frame_idx]
            if 1 not in ball_dict:
                continue

            # Use ball center in meters as reference point for proximity calculations
            ball_bbox = ball_dict[1]["bbox"]
            bx, by = self._ball_center_meters(ball_bbox)
            if not (math.isfinite(bx) and math.isfinite(by)):
                continue

            # Evaluate both outfield players and goalkeepers as valid ownership candidates
            players = {}
            players.update(tracks["players"][frame_idx])
            players.update(tracks["goalkeepers"][frame_idx])

            if not players:
                continue

            # Track the minimum distance to any candidate in this frame
            best = 1e9

            for pdata in players.values():
                px, py = self._foot_point_meters(pdata["bbox"])
                if not (math.isfinite(px) and math.isfinite(py)):
                    continue

                d = ((px - bx) ** 2 + (py - by) ** 2) ** 0.5
                if d < best:
                    best = d

            # Store only valid, finite minima to avoid contaminating calibration statistics
            if best < 1e8 and math.isfinite(best):
                distances.append(best)

        # Require a minimum sample size to avoid unstable thresholds on sparse tracking
        if len(distances) < 30:
            print("[PlayerBallAssigner] Auto-Calib: zu wenig Daten, nutze Default-Werte.")
            return

        distances = np.asarray(distances)

        # Remove outliers via median and MAD to reduce influence from mapping errors and occlusions
        med = np.nanmedian(distances)
        mad = np.nanmedian(np.abs(distances - med)) + 1e-6

        # Keep values within a robust ±3*MAD band around the median
        good_mask = (distances > med - 3 * mad) & (distances < med + 3 * mad)
        distances = distances[good_mask]

        # Re-check sample size after filtering to avoid overfitting to a tiny subset
        if len(distances) < 30:
            print("[PlayerBallAssigner] Auto-Calib: nach Outlier-Filter zu wenig Daten, nutze Default-Werte.")
            return

        # Use an upper-middle percentile to allow realistic control distance while staying conservative
        p60 = np.percentile(distances, 60)

        # Clamp the radius to a plausible range to prevent extreme calibration from bad homographies
        self.max_owner_distance_m = float(np.clip(p60, 1.0, 2.0))

        # Keep the ambiguity margin fixed as a proven stability heuristic
        self.min_margin_distance_m = 0.5

        print(
            f"[PlayerBallAssigner] Auto-Calib: "
            f"max_owner_distance_m={self.max_owner_distance_m:.2f} m, "
            f"min_margin_distance_m={self.min_margin_distance_m:.2f} m "
            f"(N={len(distances)})"
        )

    # Select the owning player by nearest distance to the ball in meters with ambiguity and radius checks
    def assign_ball_to_player(self, players, ball_bbox):
        # Convert ball position to pitch meters; return no-owner when mapping is invalid
        bx, by = self._ball_center_meters(ball_bbox)
        if not (math.isfinite(bx) and math.isfinite(by)):
            return -1

        # Track best and runner-up distances to enforce an ambiguity margin
        best_id = -1
        best_dist = 9999.0
        second_best_dist = 9999.0

        for pid, pdata in players.items():
            # Use each player's foot point as the ground-contact reference for ownership distance
            px, py = self._foot_point_meters(pdata["bbox"])
            if not (math.isfinite(px) and math.isfinite(py)):
                continue

            dist = ((px - bx) ** 2 + (py - by) ** 2) ** 0.5

            # Maintain best and second-best candidates for later ambiguity filtering
            if dist < best_dist:
                second_best_dist = best_dist
                best_dist = dist
                best_id = pid
            elif dist < second_best_dist:
                second_best_dist = dist

        # Reject ownership when the closest candidate is outside the calibrated control radius
        if best_dist > self.max_owner_distance_m:
            return -1

        # Accept the only candidate when no meaningful second-best exists
        if second_best_dist == 9999:
            return best_id

        # Accept ownership only if the best candidate is sufficiently separated from the runner-up
        if second_best_dist - best_dist >= self.min_margin_distance_m:
            return best_id

        # Return no-owner for ambiguous situations to avoid oscillating possession
        return -1