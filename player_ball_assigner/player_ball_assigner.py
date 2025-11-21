from utils import pixel_to_pitch

class PlayerBallAssigner:
    def __init__(self):
        # Schwellenwerte in Metern (nach Homographie!)
        self.max_owner_distance_m = 1.4        # < 1.4m → direkter Kontakt
        self.min_margin_distance_m = 0.5       # zweitbester Spieler mind. 0.5m weiter weg

    def _foot_point_meters(self, bbox):
        """
        Nutzt den Fußpunkt (x_center, y_bottom) und transformiert in Meter.
        """
        x1, y1, x2, y2 = bbox
        foot_x = (x1 + x2) / 2
        foot_y = y2
        return pixel_to_pitch(foot_x, foot_y)

    def _ball_center_meters(self, bbox):
        """
        Zentrum des Balles in Meter-Koordinaten.
        """
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        return pixel_to_pitch(cx, cy)

    def assign_ball_to_player(self, players, ball_bbox):
        """
        Bestimmt den Spieler, der dem Ball am nächsten ist – jetzt in Meter!

        players: dict {track_id: {"bbox": [...]}}
        ball_bbox: [x1, y1, x2, y2]
        """
        # Ballposition in Meter
        bx, by = self._ball_center_meters(ball_bbox)

        best_id = -1
        best_dist = 9999.0
        second_best_dist = 9999.0

        for pid, pdata in players.items():
            px, py = self._foot_point_meters(pdata["bbox"])

            # Distanz in Metern
            dist = ((px - bx) ** 2 + (py - by) ** 2) ** 0.5

            # Update Ranking
            if dist < best_dist:
                second_best_dist = best_dist
                best_dist = dist
                best_id = pid
            elif dist < second_best_dist:
                second_best_dist = dist

        # Falls niemand nahe genug → kein Besitzer
        if best_dist > self.max_owner_distance_m:
            return -1

        # Falls nur ein Spieler überhaupt realistisch nahe war
        if second_best_dist == 9999:
            return best_id

        # Abstand muss deutlich besser sein als der zweitbeste
        if second_best_dist - best_dist >= self.min_margin_distance_m:
            return best_id

        # Wenn beide zu nah beieinander → unsicher
        return -1