import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import math
from config import Settings

s = Settings()
analytics_cfg = s.analytics

def detect_shots(df, fps):
    """
    Erkennt Schüsse mit verbesserten Filtern gegen 'Ghost Shots'.
    """
    shots = []
    
    # --- PARAMETER (STRIKTER) ---
    min_speed = 18.0        # m/s (ca. 65 km/h) - Nur wirklich harte Aktionen
    min_cooldown = int(fps * 2.0)
    miss_tolerance = 8.0
    
    # NEU: Mindest-Flugdauer (Frames), damit es kein Jitter ist
    min_flight_frames = 5   
    # NEU: Mindest-Distanz (Meter), die der Ball zurücklegen muss
    min_shot_distance = 10.0 
    
    speed = df["ball_speed_m_s"].fillna(0).to_numpy()
    x = df["ball_x_m"].to_numpy()
    y = df["ball_y_m"].to_numpy()
    
    pitch_L = analytics_cfg.pitch_length
    pitch_W = analytics_cfg.pitch_width
    goal_center = pitch_W / 2.0
    half_goal = analytics_cfg.goal_width / 2.0
    
    valid_y_min = goal_center - half_goal - miss_tolerance
    valid_y_max = goal_center + half_goal + miss_tolerance
    goal_y_min = goal_center - half_goal
    goal_y_max = goal_center + half_goal
    
    cooldown_counter = 0
    
    # Wir iterieren durch die Frames
    for i in range(5, len(df) - 20):
        if cooldown_counter > 0:
            cooldown_counter -= 1
            continue
            
        v = speed[i]
        
        # 1. Geschwindigkeits-Check
        if v > min_speed:
            
            # NEU: Prüfen, ob die Geschwindigkeit für mind. X Frames hoch bleibt
            # Echte Schüsse bremsen nicht sofort auf 0 ab
            is_consistent = True
            for k in range(1, min_flight_frames):
                if speed[i+k] < (min_speed * 0.5): # Darf nicht sofort einbrechen
                    is_consistent = False
                    break
            if not is_consistent: continue

            # 2. Vektor berechnen (über 5 Frames für mehr Stabilität)
            dx = x[i+5] - x[i]
            dy = y[i+5] - y[i]
            
            if abs(dx) < 0.1: continue

            m = dy / dx
            shooting_team = 0
            projected_y = -1.0
            
            # Team 1 (FCH) schießt nach LINKS (auf x = 0)
            if dx < 0 and x[i] < pitch_L * 0.40:
                projected_y = y[i] + m * (0 - x[i]) 
                shooting_team = 1
                
            # Team 2 (FCE) schießt nach RECHTS (auf x = 100)
            elif dx > 0 and x[i] > pitch_L * 0.60:
                projected_y = y[i] + m * (pitch_L - x[i])
                shooting_team = 2
            
            if shooting_team == 0: continue

            # 3. Trichter-Check (Richtung Tor?)
            if not (valid_y_min <= projected_y <= valid_y_max):
                continue

            # 4. Endpunkt & Distanz bestimmen
            start_idx = i
            end_idx = i
            
            for k in range(i + 1, min(len(df), i + int(fps * 2.0))):
                # Abbruchbedingungen
                if speed[k] < 5.0 or np.isnan(x[k]):
                    end_idx = k
                    break
                # Torlinie erreicht
                if (shooting_team == 1 and x[k] <= 0) or \
                   (shooting_team == 2 and x[k] >= pitch_L):
                    end_idx = k
                    break
                end_idx = k
            
            # NEU: Distanz-Check
            # Ein Schuss muss eine gewisse Strecke fliegen
            shot_dist = math.hypot(x[end_idx] - x[start_idx], y[end_idx] - y[start_idx])
            
            if shot_dist < min_shot_distance:
                continue # War wohl nur ein kurzer Pass oder Dribbling

            # Treffer Check
            on_target = (goal_y_min <= projected_y <= goal_y_max)
            
            shots.append({
                "frame": i,
                "team": shooting_team,
                "start_x": x[i],
                "start_y": y[i],
                "end_x": 0 if shooting_team == 1 else pitch_L, 
                "end_y": projected_y,
                "speed_kmh": v * 3.6,
                "on_target": on_target
            })
            
            # Langen Cooldown setzen, um Doppelerkennung des gleichen Schusses zu vermeiden
            cooldown_counter = min_cooldown * 2 
                
    return shots

def plot_shot_map(shots, pitch_img, length, width, team1_name="Team 1", team2_name="Team 2"):
    """
    Visualisiert Schüsse auf dem Spielfeld.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(pitch_img, extent=[0, length, 0, width], alpha=0.9)
    
    # Tore einzeichnen
    goal_w = analytics_cfg.goal_width
    mid_y = width / 2
    ax.plot([0, 0], [mid_y - goal_w/2, mid_y + goal_w/2], color="white", linewidth=5, zorder=1)
    ax.plot([length, length], [mid_y - goal_w/2, mid_y + goal_w/2], color="white", linewidth=5, zorder=1)

    t1_shots = [s for s in shots if s["team"] == 1]
    t2_shots = [s for s in shots if s["team"] == 2]
    
    def draw_team_shots(shot_list, color, label):
        if not shot_list: return
        
        on_target = [s for s in shot_list if s["on_target"]]
        off_target = [s for s in shot_list if not s["on_target"]]
        
        # Treffer
        if on_target:
            x = [s["start_x"] for s in on_target]
            y = [s["start_y"] for s in on_target]
            ax.scatter(x, y, c=color, marker='o', s=120, edgecolors='white', linewidth=2, label=f"{label} (aufs Tor)", zorder=3)
            for s in on_target:
                ax.arrow(s["start_x"], s["start_y"], s["end_x"]-s["start_x"], s["end_y"]-s["start_y"], 
                         color=color, alpha=0.5, width=0.15, head_width=0.0, length_includes_head=True, zorder=2)
        
        # Daneben
        if off_target:
            x = [s["start_x"] for s in off_target]
            y = [s["start_y"] for s in off_target]
            ax.scatter(x, y, c=color, marker='x', s=80, linewidth=2, label=f"{label} (daneben)", zorder=3)
            for s in off_target:
                 ax.plot([s["start_x"], s["end_x"]], [s["start_y"], s["end_y"]], 
                         color=color, alpha=0.3, linestyle="--", linewidth=1, zorder=2)

    draw_team_shots(t1_shots, "blue", team1_name)
    draw_team_shots(t2_shots, "red", team2_name)
    
    ax.set_title(f"Shot Map\n{team1_name} vs {team2_name}")
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    ax.set_xlim(0, length)
    ax.set_ylim(0, width)
    ax.axis("off")
    
    plt.tight_layout()
    return fig