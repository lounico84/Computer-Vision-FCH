import numpy as np

from config import Settings
from player_ball_assigner import PlayerBallAssigner

# Compute team ball posession over all frames
def compute_team_ball_control(tracks, settings: Settings):

    player_assigner = PlayerBallAssigner()
    ball_cfg = settings.ball_control

    num_frames = len(tracks["players"])
    team_ball_control = []

    # --- Team-Hysterese (wie bisher) ---
    last_team = 0           # last confirmed team in control
    candidate_team = None   # potential new team taking over
    candidate_count = 0     # how many consecutive frames this new team appears

    # --- NEU: Besitzer-Hysterese ---
    last_owner_id = None        # stabiler Besitzer (track_id)
    owner_candidate_id = None   # Kandidat für neuen Besitzer
    owner_candidate_count = 0   # wie viele Frames in Folge dieser Kandidat vorkommt

    for frame_idx in range(num_frames):
        players = tracks["players"][frame_idx]
        goalkeepers = tracks["goalkeepers"][frame_idx]
        ball_dict = tracks["ball"][frame_idx]

        # Default: kein Besitzer für diesen Frame
        owner_id = -1
        owner_team = 0

        # ==== Fall 1: Ball in diesem Frame gar nicht vorhanden ====
        if 1 not in ball_dict:
            # kein Ball -> kein Besitzer, kein Team in Kontrolle
            team_ball_control.append(0)
            continue

        # Ab hier wissen wir: ball_dict[1] existiert
        ball_bbox = ball_dict[1]["bbox"]

        # Merge players and goalkeepers into one dictionary
        all_actors = {}
        all_actors.update(players)
        all_actors.update(goalkeepers)

        # Roher Besitzer (nächster Spieler zum Ball)
        assigned_id = player_assigner.assign_ball_to_player(all_actors, ball_bbox)

        # ==== Fall 2: gültiger Ball, aber kein passender Spieler ====
        if assigned_id == -1:
            # Ball ist da, aber keinem Spieler eindeutig zuordenbar
            # -> kein Besitzer, keine klare Kontrolle
            ball_dict[1]["owner_id"] = owner_id
            ball_dict[1]["owner_team"] = owner_team
            team_ball_control.append(0)
            continue

        # ==== Besitzer-Hysterese: stabilen owner_id bestimmen ====
        if last_owner_id is None:
            # Erster gültiger Besitzer
            last_owner_id = assigned_id
            owner_candidate_id = None
            owner_candidate_count = 0
        elif assigned_id == last_owner_id:
            # gleicher Besitzer wie bisher -> Kandidat zurücksetzen
            owner_candidate_id = None
            owner_candidate_count = 0
        else:
            # anderer Spieler als bisheriger Besitzer
            if owner_candidate_id == assigned_id:
                owner_candidate_count += 1
            else:
                owner_candidate_id = assigned_id
                owner_candidate_count = 1

            # Wechsel nur, wenn der Kandidat stabil genug ist
            if owner_candidate_count >= ball_cfg.min_switch_frames:
                last_owner_id = assigned_id
                owner_candidate_id = None
                owner_candidate_count = 0

        owner_id = last_owner_id

        # ==== Besitzer im aktuellen Frame verankern ====
        raw_team = 0
        if owner_id in players:
            players[owner_id]["has_ball"] = True
            raw_team = players[owner_id].get("team", 0)
        elif owner_id in goalkeepers:
            goalkeepers[owner_id]["has_ball"] = True
            raw_team = goalkeepers[owner_id].get("team", 0)
        else:
            # stabiler Besitzer existiert, ist aber in diesem Frame nicht sichtbar
            # -> Teamkontrolle bleibt wie bisher
            ball_dict[1]["owner_id"] = owner_id
            ball_dict[1]["owner_team"] = 0
            team_ball_control.append(last_team)
            continue

        owner_team = raw_team

        # Besitzer-Info am Ball speichern (für spätere Auswertungen)
        ball_dict[1]["owner_id"] = owner_id
        ball_dict[1]["owner_team"] = owner_team

        # ==== Team-Kontrolle bestimmen ====
        if raw_team not in (1, 2):
            # kein valides Team erkannt
            team_ball_control.append(last_team if last_team != 0 else 0)
            continue

        # Team-Hysterese wie bisher
        if last_team == 0:
            last_team = raw_team
            candidate_team = None
            candidate_count = 0
        elif raw_team == last_team:
            candidate_team = None
            candidate_count = 0
        else:
            if candidate_team == raw_team:
                candidate_count += 1
            else:
                candidate_team = raw_team
                candidate_count = 1

            if candidate_count >= ball_cfg.min_switch_frames:
                last_team = raw_team
                candidate_team = None
                candidate_count = 0

        team_ball_control.append(last_team)

    return np.array(team_ball_control)