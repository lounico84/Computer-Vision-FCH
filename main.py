import cv2
import numpy as np
from utils import read_video, save_video, get_center_of_bbox
from trackers import Tracker
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner

def main():
    # Read Video
    #video_frames = read_video('input_videos_match/Test/wiesendangen_test_clip_short.mp4')
    video_frames = read_video('input_videos_match/Test/kuesnacht_test_clip2.MP4')

    # Initialize Tracker
    tracker =  Tracker('yolo_training/models/fifth_model/run1/weights/best.pt')

    tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path='project/Computer-Vision-FCH/stubs/track_stubs_k.pkl')
    '''
    # Save cropped image

    frame_idx = 1
    track_id = 17 

    referee = tracks['referees'][frame_idx][track_id]
    bbox = referee['bbox']
    frame = video_frames[frame_idx]

    x1, y1, x2, y2 = map(int, bbox)
    cropped_image = frame[y1:y2, x1:x2]

    cv2.imwrite("output_video_match/cropped_player_19.jpg", cropped_image)
    exit()
    '''

    # Intepolate Ball Positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # Assign Player Team
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames, tracks)
    team_assigner.assign_referee_color(video_frames, tracks)
    team_assigner.save_color_debug("output_video_match/color_debug_k.png")
    
    # 1. PASS: nur Farben/Abstände messen und Votes sammeln
    for frame_number, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team_assigner.get_player_team(
                video_frames[frame_number],
                track['bbox'],
                player_id
            )
    
    # Nach dem 1. PASS: Mehrheits-Team pro Track bestimmen
    for player_id, votes_dict in team_assigner.team_vote_counts.items():
        v1 = votes_dict.get(1, 0)
        v2 = votes_dict.get(2, 0)
        # einfache Mehrheit; bei Gleichstand Team 1
        majority_team = 1 if v1 >= v2 else 2
        team_assigner.player_team_dict[player_id] = majority_team

    # Finale Entscheidung pro Track: Ref oder Team 1/2
    final_labels = {}  # player_id -> 0,1,2

    min_votes = 7
    min_ratio = 0.8
    min_observations = 10

    for player_id, obs in team_assigner.player_obs_counts.items():
        votes = team_assigner.ref_vote_counts.get(player_id, 0)

        # Default-Team jetzt: Mehrheits-Team aus oben
        default_team = team_assigner.player_team_dict.get(player_id, 1)

        if (
            team_assigner.referee_color is not None
            and obs >= min_observations
            and votes >= min_votes
            and votes >= min_ratio * obs
        ):
            final_labels[player_id] = 0   # Ref
        else:
            final_labels[player_id] = default_team

    # 2. PASS: Tracks anhand finaler Labels umbauen
    for frame_number, player_track in enumerate(tracks['players']):
        for player_id, track in list(player_track.items()):
            team = final_labels.get(player_id, 1)

            if team == 0:
                # kompletter Track ist Schiri/Linienrichter
                tracks['referees'][frame_number][player_id] = track
                del tracks['players'][frame_number][player_id]
            else:
                tracks['players'][frame_number][player_id]['team'] = team
                tracks['players'][frame_number][player_id]['team_color'] = team_assigner.team_colors[team]

    # --- Team-Schwerpunkte (x-Position) aus Spieler-Tracks bestimmen ---
    team_x_positions = {1: [], 2: []}

    for frame_number, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = track.get('team')
            if team not in (1, 2):
                continue
            x, _ = get_center_of_bbox(track['bbox'])
            team_x_positions[team].append(x)

    team_mean_x = {}
    for t in (1, 2):
        if len(team_x_positions[t]) > 0:
            team_mean_x[t] = float(np.mean(team_x_positions[t]))
        else:
            team_mean_x[t] = None  # falls ein Team nie gesehen wurde (eher unwahrscheinlich)
    
    # --- Durchschnitts-x pro Torhüter-Track ---
    goalkeeper_x_positions = {}  # gk_id -> Liste von x

    for frame_number, gk_track in enumerate(tracks['goalkeepers']):
        for gk_id, track in gk_track.items():
            x, _ = get_center_of_bbox(track['bbox'])
            if gk_id not in goalkeeper_x_positions:
                goalkeeper_x_positions[gk_id] = []
            goalkeeper_x_positions[gk_id].append(x)

    gk_mean_x = {gk_id: float(np.mean(xs)) for gk_id, xs in goalkeeper_x_positions.items()}

        # --- Torhüter -> Team-Zuordnung nach räumlicher Nähe zu Team-Schwerpunkten ---
    goalkeeper_team_map = {}

    for gk_id, gk_x in gk_mean_x.items():
        best_team = None
        best_dist = float('inf')

        for t in (1, 2):
            if team_mean_x[t] is None:
                continue
            dist = abs(gk_x - team_mean_x[t])
            if dist < best_dist:
                best_dist = dist
                best_team = t

        # Fallback, falls aus irgendeinem Grund beide None sind
        if best_team is None:
            best_team = 1

        goalkeeper_team_map[gk_id] = best_team
    
        # --- Keeper in den Tracks mit Team versehen (Farbe bleibt blau) ---
    for frame_number, gk_track in enumerate(tracks['goalkeepers']):
        for gk_id, track in gk_track.items():
            team = goalkeeper_team_map.get(gk_id, 1)
            tracks['goalkeepers'][frame_number][gk_id]['team'] = team
            # kein 'team_color' setzen, damit draw_annotations weiterhin Blau benutzt

    # Assign Ball Acquisition (inkl. Torhüter) + geglätteter Team-Ballbesitz
    player_assigner = PlayerBallAssigner()
    team_ball_control = []

    num_frames = len(tracks['players'])

    last_team = 0           # aktuelles „offizielles“ Ballbesitz-Team (1/2 oder 0)
    candidate_team = None   # Team, das evtl. übernehmen will
    candidate_count = 0     # wie viele Frames hintereinander dieser Kandidat vorne ist
    min_switch_frames = 5   # Mindestanzahl Frames, bevor der Besitz offiziell wechselt

    for frame_num in range(num_frames):
        player_track = tracks['players'][frame_num]
        gk_track     = tracks['goalkeepers'][frame_num]
        ball_dict    = tracks['ball'][frame_num]

        # --- 1) Ball sichtbar? ---
        if 1 not in ball_dict:
            # Kein Ball -> keine neuen Infos, Frame zählt als „kein klarer Besitz“
            team_ball_control.append(0)
            continue

        ball_bbox = ball_dict[1]['bbox']

        # --- 2) Spieler & Keeper zusammen betrachten ---
        all_actors = {}
        all_actors.update(player_track)
        all_actors.update(gk_track)

        assigned_id = player_assigner.assign_ball_to_player(all_actors, ball_bbox)

        # --- 3) eventuell niemand klar genug in der Nähe ---
        if assigned_id == -1:
            team_ball_control.append(0)
            continue

        # --- 4) Team des „rohen“ Besitzers bestimmen + has_ball setzen ---
        if assigned_id in player_track:
            player_track[assigned_id]['has_ball'] = True
            raw_team = player_track[assigned_id]['team']
        else:
            gk_track[assigned_id]['has_ball'] = True
            raw_team = gk_track[assigned_id]['team']

        if raw_team not in (1, 2):
            team_ball_control.append(0)
            continue

        # --- 5) Hysterese: Wechsel des Ballbesitz-Teams glätten ---
        if last_team == 0:
            # Noch kein offizieller Besitz -> direkt übernehmen
            last_team = raw_team
            candidate_team = None
            candidate_count = 0
        else:
            if raw_team == last_team:
                # Bestätigung des aktuellen Teams
                candidate_team = None
                candidate_count = 0
            else:
                # anderes Team versucht zu übernehmen
                if candidate_team == raw_team:
                    candidate_count += 1
                else:
                    candidate_team = raw_team
                    candidate_count = 1

                # Wenn der Kandidat mehrere Frames hintereinander vorne ist -> Besitzwechsel
                if candidate_count >= min_switch_frames:
                    last_team = raw_team
                    candidate_team = None
                    candidate_count = 0

        # last_team ist das geglättete Anzeige-Team
        team_ball_control.append(last_team)

    team_ball_control = np.array(team_ball_control)
    
    # Draw Output
    ## Draw object tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)

    # Save Video
    save_video(output_video_frames, 'output_video_match/output_video_k.avi')

if __name__ == '__main__':
    main()