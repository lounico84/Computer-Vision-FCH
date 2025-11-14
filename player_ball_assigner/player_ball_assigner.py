from utils import get_center_of_bbox, measure_distance

class PlayerBallAssigner():
    def __init__(self):
        # Maximum distance in pixels at wich a player can still be considered a valid candidate for owning the ball
        self.max_player_ball_distance = 70

        # Rule the closest player must be clearly better compared to the second closest candidate
        self.min_margin_ratio = 0.7 
    
    # Assign the ball to the closest player or goalkeeper
    def assign_ball_to_player(self, players, ball_bbox):

        # Compute the (x,y) center position of the ball
        ball_position = get_center_of_bbox(ball_bbox)

        best_id = -1
        best_dist = 99999 # current smallest distance
        second_best_dist = 99999 # second smallest distance

        # Loop through all players and goalkeepers
        for player_id, player in players.items():
            player_bbox = player['bbox']

            # Compute distances from both left and right foot positions to the ball
            # Using min distance increases robustness to slight bbox noise
            distance_left = measure_distance((player_bbox[0], player_bbox[-1]), ball_position)
            distance_right = measure_distance((player_bbox[2], player_bbox[-1]), ball_position)
            distance = min(distance_left, distance_right)

            # Update best and second best distances
            if distance < best_dist:
                second_best_dist = best_dist
                best_dist = distance
                best_id = player_id
            elif distance < second_best_dist:
                second_best_dist = distance

        # No valid candidate if the closest plaxer is still to far away
        if best_dist > self.max_player_ball_distance:
            return -1

        # If only one player was close enough
        if second_best_dist == 99999:
            return best_id

        # Rule: the best distance must be clearly better than the second best tp avoid false assignments when multiple plaxers are equally close
        if best_dist < self.min_margin_ratio * second_best_dist:
            return best_id
        else:
            return -1