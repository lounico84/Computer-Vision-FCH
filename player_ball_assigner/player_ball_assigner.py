import math
import numpy as np

from utils import pixel_to_pitch

class PlayerBallAssigner:
    def __init__(self):
        self.max_owner_distance_m = 1.4
        self.min_margin_distance_m = 0.5

    def _foot_point_meters(self, bbox):
        x1, y1, x2, y2 = bbox
        foot_x = (x1 + x2) / 2
        foot_y = y2
        return pixel_to_pitch(foot_x, foot_y)

    def _ball_center_meters(self, bbox):
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        return pixel_to_pitch(cx, cy)

    def auto_calibrate_from_tracks(self, tracks, max_frames: int = 600):
        """
        Schätzt sinnvolle Distanzen aus den ersten max_frames Frames.
        Nutzt NUR die Geometrie der aktuellen Homographie.
        """
        distances = []
        diffs_best_second = []

        num_frames = min(max_frames, len(tracks["ball"]))

        for frame_idx in range(num_frames):
            ball_dict = tracks["ball"][frame_idx]
            if 1 not in ball_dict:
                continue

            ball_bbox = ball_dict[1]["bbox"]
            bx, by = self._ball_center_meters(ball_bbox)
            if not (math.isfinite(bx) and math.isfinite(by)):
                continue

            # Spieler + Torhüter
            players = {}
            players.update(tracks["players"][frame_idx])
            players.update(tracks["goalkeepers"][frame_idx])

            if not players:
                continue

            best = 1e9
            second = 1e9

            for pdata in players.values():
                px, py = self._foot_point_meters(pdata["bbox"])
                if not (math.isfinite(px) and math.isfinite(py)):
                    continue

                d = ((px - bx) ** 2 + (py - by) ** 2) ** 0.5
                if d < best:
                    second = best
                    best = d
                elif d < second:
                    second = d

            if best < 1e8 and math.isfinite(best):
                distances.append(best)
                if second < 1e8 and math.isfinite(second):
                    diffs_best_second.append(second - best)

        if len(distances) < 30:
            print("[PlayerBallAssigner] Auto-Calib: zu wenig Daten, nutze Default-Werte.")
            return

        distances = np.asarray(distances)
        if diffs_best_second:
            diffs = np.asarray(diffs_best_second)
        else:
            diffs = None

        # Typische Spieler-Ball-Distanz ~ Median/Perzentil
        p50 = np.percentile(distances, 50)
        p80 = np.percentile(distances, 80)

        # z.B. zwischen Median und 80%-Perzentil
        self.max_owner_distance_m = float(p80)

        if diffs is not None and len(diffs) >= 10:
            dmed = np.percentile(diffs, 50)
            self.min_margin_distance_m = float(max(0.2, dmed))
        else:
            self.min_margin_distance_m = 0.4

        print(
            f"[PlayerBallAssigner] Auto-Calib: "
            f"max_owner_distance_m={self.max_owner_distance_m:.2f} m, "
            f"min_margin_distance_m={self.min_margin_distance_m:.2f} m "
            f"(N={len(distances)})"
        )

    def assign_ball_to_player(self, players, ball_bbox):
        bx, by = self._ball_center_meters(ball_bbox)
        if not (math.isfinite(bx) and math.isfinite(by)):
            return -1

        best_id = -1
        best_dist = 9999.0
        second_best_dist = 9999.0

        for pid, pdata in players.items():
            px, py = self._foot_point_meters(pdata["bbox"])
            if not (math.isfinite(px) and math.isfinite(py)):
                continue

            dist = ((px - bx) ** 2 + (py - by) ** 2) ** 0.5

            if dist < best_dist:
                second_best_dist = best_dist
                best_dist = dist
                best_id = pid
            elif dist < second_best_dist:
                second_best_dist = dist

        if best_dist > self.max_owner_distance_m:
            return -1

        if second_best_dist == 9999:
            return best_id

        if second_best_dist - best_dist >= self.min_margin_distance_m:
            return best_id

        return -1