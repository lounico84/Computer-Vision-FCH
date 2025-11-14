from utils import get_center_of_bbox, measure_distance

class PlayerBallAssigner():
    def __init__(self):
        self.max_player_ball_distance = 70
        self.min_margin_ratio = 0.7   # bester Abstand muss z.B. < 70% des zweitbesten sein
    
    def assign_ball_to_player(self, players, ball_bbox):
        ball_position = get_center_of_bbox(ball_bbox)

        best_id = -1
        best_dist = 99999
        second_best_dist = 99999

        for player_id, player in players.items():
            player_bbox = player['bbox']

            distance_left = measure_distance((player_bbox[0], player_bbox[-1]), ball_position)
            distance_right = measure_distance((player_bbox[2], player_bbox[-1]), ball_position)
            distance = min(distance_left, distance_right)

            if distance < best_dist:
                second_best_dist = best_dist
                best_dist = distance
                best_id = player_id
            elif distance < second_best_dist:
                second_best_dist = distance

        # keine Kandidaten im sinnvollen Bereich
        if best_dist > self.max_player_ball_distance:
            return -1

        # wenn es keinen zweitbesten gibt (nur ein Spieler in der Nähe)
        if second_best_dist == 99999:
            return best_id

        # Sicherheitsmarge: bester Abstand muss deutlich besser sein als der zweitbeste
        if best_dist < self.min_margin_ratio * second_best_dist:
            return best_id
        else:
            return -1