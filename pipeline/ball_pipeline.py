import numpy as np

from config import Settings
from player_ball_assigner import PlayerBallAssigner

# Compute team ball possession over all frames
def compute_team_ball_control(tracks, settings: Settings):

    player_assigner = PlayerBallAssigner()
    ball_cfg = settings.ball_control

    num_frames = len(tracks["players"])
    team_ball_control = []

    # --- Team-Hysterese wie bisher ---
    last_team = 0           # letzter sicherer Ballbesitz (Team 1 oder 2)
    candidate_team = None   # potenziell neues Team
    candidate_count = 0     # wie viele Frames hintereinander dieses neue Team

    # --- Besitzer-Hysterese ---
    last_owner_id = None        # stabiler Besitzer (track_id)
    owner_candidate_id = None   # Kandidat für neuen Besitzer
    owner_candidate_count = 0   # wie viele Frames in Folge dieser Kandidat

    # Wie lange darf der Ball "frei" sein, ohne dass wir die Teamkontrolle verlieren?
    # z.B. 10 Frames bei 30 fps ≈ 0.33 Sekunden – typischer Flugpass/Chip
    max_free_ball_frames = 10
    free_ball_counter = 0

    for frame_idx in range(num_frames):
        players = tracks["players"][frame_idx]
        goalkeepers = tracks["goalkeepers"][frame_idx]
        ball_dict = tracks["ball"][frame_idx]

        # Default: kein Besitzer für diesen Frame
        owner_id = -1
        owner_team = 0

        # ==== Fall 1: Ball in diesem Frame gar nicht vorhanden ====
        if 1 not in ball_dict:
            # Ball ist nicht sichtbar -> wie "Ball frei"
            free_ball_counter += 1

            # Besitzer-Info am Ball (zur Sicherheit auf -1 setzen)
            # (Ball existiert hier ja gar nicht in ball_dict, daher kein write)
            # Teamkontrolle: solange free_ball_counter klein ist, bleibt last_team
            if last_team != 0 and free_ball_counter <= max_free_ball_frames:
                team_ball_control.append(last_team)
            else:
                team_ball_control.append(0)

            continue

        # Ab hier wissen wir: ball_dict[1] existiert
        ball_bbox = ball_dict[1]["bbox"]

        # Merge players and goalkeepers into one dictionary
        # WICHTIG: keine Referees – die dürfen nie Besitzer sein
        all_actors = {}
        all_actors.update(players)
        all_actors.update(goalkeepers)

        # Roher Besitzer (nächster Spieler/Goalie zum Ball – bereits in Metern)
        assigned_id = player_assigner.assign_ball_to_player(all_actors, ball_bbox)

        # ==== Fall 2: gültiger Ball, aber kein passender Spieler ====
        if assigned_id == -1:
            # Ball ist sichtbar, aber keinem Spieler eindeutig zuordenbar -> Ball "frei"
            free_ball_counter += 1

            ball_dict[1]["owner_id"] = owner_id
            ball_dict[1]["owner_team"] = owner_team

            # Teamkontrolle beibehalten, solange der Ball nur kurz frei ist
            if last_team != 0 and free_ball_counter <= max_free_ball_frames:
                team_ball_control.append(last_team)
            else:
                team_ball_control.append(0)

            continue

        # Ab hier: ein Kandidat für Besitzer existiert -> Ball ist NICHT mehr frei
        free_ball_counter = 0

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

            # wenn wir schon ein last_team haben, behalten wir es
            team_ball_control.append(last_team)
            continue

        owner_team = raw_team

        # Besitzer-Info am Ball speichern (für spätere Auswertungen / CSV)
        ball_dict[1]["owner_id"] = owner_id
        ball_dict[1]["owner_team"] = owner_team

        # ==== Team-Kontrolle bestimmen ====
        if raw_team not in (1, 2):
            # kein valides Team erkannt
            team_ball_control.append(last_team if last_team != 0 else 0)
            continue

        # Team-Hysterese wie bisher, aber robuster
        if last_team == 0:
            # erstes Mal ein Team in Kontrolle
            last_team = raw_team
            candidate_team = None
            candidate_count = 0
        elif raw_team == last_team:
            # gleiches Team wie bisher -> Kandidat zurücksetzen
            candidate_team = None
            candidate_count = 0
        else:
            # mögliches neues Team
            if candidate_team == raw_team:
                candidate_count += 1
            else:
                candidate_team = raw_team
                candidate_count = 1

            # nur wenn neuer Kandidat stabil genug ist -> Teamwechsel
            if candidate_count >= ball_cfg.min_switch_frames:
                last_team = raw_team
                candidate_team = None
                candidate_count = 0

        team_ball_control.append(last_team)

    return np.array(team_ball_control)