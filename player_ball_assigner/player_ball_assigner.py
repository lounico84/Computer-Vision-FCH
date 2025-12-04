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

    def auto_calibrate_from_tracks(self, tracks, max_frames: int = 2000):
        """
        Schätzt sinnvolle Distanzen aus den ersten max_frames Frames.
        Nutzt NUR die Geometrie der aktuellen Homographie.

        Idee:
        - Alle Ball–Spieler-Minimaldistanzen sammeln.
        - Grobe Ausreißer über robusten MAD-Filter entfernen.
        - max_owner_distance_m als Perzentil der gefilterten Distanzen setzen,
          aber auf einen realistischen Bereich clippen.
        - min_margin_distance_m fix auf 0.5 lassen.
        """
        distances = []

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

            for pdata in players.values():
                px, py = self._foot_point_meters(pdata["bbox"])
                if not (math.isfinite(px) and math.isfinite(py)):
                    continue

                d = ((px - bx) ** 2 + (py - by) ** 2) ** 0.5
                if d < best:
                    best = d

            if best < 1e8 and math.isfinite(best):
                distances.append(best)

        if len(distances) < 30:
            print("[PlayerBallAssigner] Auto-Calib: zu wenig Daten, nutze Default-Werte.")
            return

        distances = np.asarray(distances)

        # --- robuste Ausreißer-Filterung (Median + MAD) ---
        med = np.nanmedian(distances)
        mad = np.nanmedian(np.abs(distances - med)) + 1e-6  # Robustheit

        # alles innerhalb von ±3*MAD behalten
        good_mask = (distances > med - 3 * mad) & (distances < med + 3 * mad)
        distances = distances[good_mask]

        if len(distances) < 30:
            print("[PlayerBallAssigner] Auto-Calib: nach Outlier-Filter zu wenig Daten, nutze Default-Werte.")
            return

        # Typische Distanz: eher im unteren Bereich der Verteilung
        p60 = np.percentile(distances, 60)

        # Realistischer Bereich begrenzen (z.B. 1.0–2.0 m)
        self.max_owner_distance_m = float(np.clip(p60, 1.0, 2.0))

        # Bewährter fixer Wert
        self.min_margin_distance_m = 0.5

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