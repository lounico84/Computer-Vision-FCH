import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import math
from mplsoccer import Pitch
from config import Settings

s = Settings()
analytics_cfg = s.analytics


# Detect shot events from high ball-speed bursts plus goal-directed trajectory and distance consistency checks
def detect_shots(df, fps):
    shots = []
    
    # Configure strict thresholds to reduce false positives from jitter and short actions
    min_speed = 18.0        # m/s (ca. 65 km/h) - Nur wirklich harte Aktionen
    min_cooldown = int(fps * 2.0)
    miss_tolerance = 8.0
    
    # Enforce minimum flight duration to filter single-frame spikes
    min_flight_frames = 5   
    # Enforce minimum travel distance to separate shots from short passes/dribbles
    min_shot_distance = 10.0 
    
    speed = df["ball_speed_m_s"].fillna(0).to_numpy()
    x = df["ball_x_m"].to_numpy()
    y = df["ball_y_m"].to_numpy()
    
    pitch_L = analytics_cfg.pitch_length
    pitch_W = analytics_cfg.pitch_width
    goal_center = pitch_W / 2.0
    half_goal = analytics_cfg.goal_width / 2.0
    
    # Define target corridor (with tolerance) and true goal mouth bounds
    valid_y_min = goal_center - half_goal - miss_tolerance
    valid_y_max = goal_center + half_goal + miss_tolerance
    goal_y_min = goal_center - half_goal
    goal_y_max = goal_center + half_goal
    
    cooldown_counter = 0
    
    # Iterate across frames with margins for lookahead and end-point search
    for i in range(5, len(df) - 20):
        if cooldown_counter > 0:
            cooldown_counter -= 1
            continue
            
        v = speed[i]
        
        # Trigger candidate shot detection only for high-speed ball actions
        if v > min_speed:
            
            # Validate sustained speed over several frames to avoid "ghost" spikes
            is_consistent = True
            for k in range(1, min_flight_frames):
                if speed[i+k] < (min_speed * 0.5):
                    is_consistent = False
                    break
            if not is_consistent: continue

            # Estimate movement vector over multiple frames for a more stable direction
            dx = x[i+5] - x[i]
            dy = y[i+5] - y[i]
            
            if abs(dx) < 0.1: continue

            m = dy / dx
            shooting_team = 0
            projected_y = -1.0
            
            # Infer shooting direction and team based on field position and x-axis travel
            if dx < 0 and x[i] < pitch_L * 0.40:
                projected_y = y[i] + m * (0 - x[i]) 
                shooting_team = 1
                
            elif dx > 0 and x[i] > pitch_L * 0.60:
                projected_y = y[i] + m * (pitch_L - x[i])
                shooting_team = 2
            
            if shooting_team == 0: continue

            # Require goal-directed corridor to filter lateral high-speed events
            if not (valid_y_min <= projected_y <= valid_y_max):
                continue

            # Find an approximate end frame based on deceleration, missing tracking, or goal line contact
            start_idx = i
            end_idx = i
            
            for k in range(i + 1, min(len(df), i + int(fps * 2.0))):
                if speed[k] < 5.0 or np.isnan(x[k]):
                    end_idx = k
                    break
                if (shooting_team == 1 and x[k] <= 0) or \
                   (shooting_team == 2 and x[k] >= pitch_L):
                    end_idx = k
                    break
                end_idx = k
            
            # Enforce minimum travel distance to avoid classifying short actions as shots
            shot_dist = math.hypot(x[end_idx] - x[start_idx], y[end_idx] - y[start_idx])
            if shot_dist < min_shot_distance:
                continue

            # Flag shots on target based on whether the projection falls inside the goal mouth
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
            
            # Apply a strong cooldown to prevent double counting the same shot sequence
            cooldown_counter = min_cooldown * 2 
                
    return shots


# Plot a dark-mode shot map with team colors and separate styling for on-target vs off-target attempts
def plot_shot_map(shots, pitch_img, length, width, team1_name="Team 1", team2_name="Team 2"):
    # Create a custom-dimension pitch to match the coordinate system of the event data
    pitch = Pitch(
        pitch_type='custom',
        pitch_length=length,
        pitch_width=width,
        line_color='#c7d5cc',
        pitch_color='#22312b',
        linewidth=2,
        goal_type='box'
    )

    # Create a standalone figure for export-ready visualization
    fig, ax = pitch.draw(figsize=(10, 6))
    fig.set_facecolor('#22312b')

    # Standardize orientation to match other pitch analytics
    ax.invert_yaxis()

    # Split shots by team for consistent color coding and labeling
    t1_shots = [s for s in shots if s["team"] == 1]
    t2_shots = [s for s in shots if s["team"] == 2]
    
    c1 = '#00bfff' # Team 1
    c2 = '#dc143c' # Team 2

    # Encapsulate per-team rendering logic to keep styling consistent across teams
    def draw_team_shots(shot_list, color, label):
        if not shot_list: return
        
        on_target = [s for s in shot_list if s["on_target"]]
        off_target = [s for s in shot_list if not s["on_target"]]
        
        # Render on-target shots with full trajectory arrows and launch markers
        if on_target:
            pitch.arrows(
                [s["start_x"] for s in on_target], [s["start_y"] for s in on_target],
                [s["end_x"] for s in on_target], [s["end_y"] for s in on_target],
                width=2, headwidth=3, headlength=3, 
                color=color, ax=ax, label=f"{label} (On Target)", zorder=3
            )
            pitch.scatter(
                [s["start_x"] for s in on_target], [s["start_y"] for s in on_target],
                s=80, color=color, edgecolors='white', linewidth=1.5, ax=ax, zorder=4
            )
        
        # Render off-target shots with reduced emphasis to avoid visual overload
        if off_target:
            pitch.scatter(
                [s["start_x"] for s in off_target], [s["start_y"] for s in off_target],
                marker='x', s=60, color=color, alpha=0.7, ax=ax, label=f"{label} (Off Target)", zorder=2
            )
            pitch.lines(
                [s["start_x"] for s in off_target], [s["start_y"] for s in off_target],
                [s["end_x"] for s in off_target], [s["end_y"] for s in off_target],
                color=color, alpha=0.3, lw=1, ls='--', ax=ax, zorder=1
            )

    # Render both teams on the same pitch for immediate comparison
    draw_team_shots(t1_shots, c1, team1_name)
    draw_team_shots(t2_shots, c2, team2_name)
    
    # Add an export-friendly title centered above the pitch
    fig.text(0.5, 0.95, f"Shot Map: {team1_name} vs {team2_name}", 
             color='white', fontsize=18, fontweight='bold', 
             ha='center', va='center')
    
    # Provide a compact legend tuned for dark backgrounds
    legend = ax.legend(facecolor='#22312b', edgecolor='None', fontsize=9, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.05))
    for text in legend.get_texts():
        text.set_color("white")
    
    plt.tight_layout()
    return fig