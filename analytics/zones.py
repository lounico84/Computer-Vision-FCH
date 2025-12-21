import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from config import Settings

# Provide a safe fallback when mplsoccer is not installed (keeps plotting pipeline callable)
try:
    from mplsoccer import Pitch
except ImportError:
    class Pitch:
        def __init__(self, *args, **kwargs): pass
        def draw(self, *args, **kwargs): return plt.subplots()
        def bin_statistic(self, *args, **kwargs): return {}
        def heatmap(self, *args, **kwargs): pass

s = Settings()


# Build a 3-panel dashboard: attack side preference, zone dominance heatmap, and third-based possession distribution
def plot_zone_analysis(df, pitch_length, pitch_width, team1_name="Team 1", team2_name="Team 2"):   
    # Create a 3-column layout for a single, slide-ready tactical dashboard
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.set_facecolor('#22312b')
    
    # Initialize a custom pitch matching the coordinate system used in the event data
    pitch = Pitch(
        pitch_type='custom', pitch_length=pitch_length, pitch_width=pitch_width,
        line_color='#c7d5cc', pitch_color='#22312b', linewidth=2,
    )

    ax1 = axes[0]
    ax1.set_facecolor('#22312b')
    pitch.draw(ax=ax1)
    ax1.invert_yaxis()
    
    # Filter ball-control samples restricted to each team's attacking third
    t1_att = df[(df["team_ball_control"] == 1) & (df["ball_x_m"] < pitch_length / 3)]
    t2_att = df[(df["team_ball_control"] == 2) & (df["ball_x_m"] > pitch_length * 2/3)]
    
    # Convert top/mid/bottom field usage into left/center/right from the team's playing perspective
    def get_flank_pct(data, team_direction):
        if len(data) == 0: return [0, 0, 0]
        
        top = len(data[data["ball_y_m"] < pitch_width/3])
        mid = len(data[(data["ball_y_m"] >= pitch_width/3) & (data["ball_y_m"] <= pitch_width*2/3)])
        bot = len(data[data["ball_y_m"] > pitch_width*2/3])
        total = top + mid + bot
        
        if team_direction == "R_to_L":
            return [bot/total*100, mid/total*100, top/total*100]
        else:
            return [top/total*100, mid/total*100, bot/total*100]

    # Compute left/center/right attack shares for both teams in their respective final third
    pct1 = get_flank_pct(t1_att, "R_to_L")
    pct2 = get_flank_pct(t2_att, "L_to_R")
    
    # Render Team 1 flank shares as horizontal bars on the left side of the pitch
    ax1.barh(pitch_width*5/6, pct1[0]/3, left=5, height=6, color='#00bfff', alpha=0.8)
    ax1.text(5 + pct1[0]/3 + 2, pitch_width*5/6, f"{int(pct1[0])}%", color='#00bfff', va='center', fontweight='bold')
    
    ax1.barh(pitch_width/2, pct1[1]/3, left=5, height=6, color='#00bfff', alpha=0.8)
    ax1.text(5 + pct1[1]/3 + 2, pitch_width/2, f"{int(pct1[1])}%", color='#00bfff', va='center', fontweight='bold')

    ax1.barh(pitch_width/6, pct1[2]/3, left=5, height=6, color='#00bfff', alpha=0.8)
    ax1.text(5 + pct1[2]/3 + 2, pitch_width/6, f"{int(pct1[2])}%", color='#00bfff', va='center', fontweight='bold')

    # Render Team 2 flank shares as horizontal bars on the right side (negative width for leftward growth)
    ax1.barh(pitch_width/6, -pct2[0]/3, left=95, height=6, color='#dc143c', alpha=0.8)
    ax1.text(95 - pct2[0]/3 - 8, pitch_width/6, f"{int(pct2[0])}%", color='#dc143c', va='center', fontweight='bold')

    ax1.barh(pitch_width/2, -pct2[1]/3, left=95, height=6, color='#dc143c', alpha=0.8)
    ax1.text(95 - pct2[1]/3 - 8, pitch_width/2, f"{int(pct2[1])}%", color='#dc143c', va='center', fontweight='bold')
    
    ax1.barh(pitch_width*5/6, -pct2[2]/3, left=95, height=6, color='#dc143c', alpha=0.8)
    ax1.text(95 - pct2[2]/3 - 8, pitch_width*5/6, f"{int(pct2[2])}%", color='#dc143c', va='center', fontweight='bold')

    # Provide a clear panel title and directional hints for interpretation
    ax1.set_title("Attack Sides (Final Third)", color='white', fontsize=14, fontweight='bold')
    ax1.arrow(40, pitch_width/2, -10, 0, head_width=3, color='#00bfff', alpha=0.5)
    ax1.arrow(60, pitch_width/2, 10, 0, head_width=3, color='#dc143c', alpha=0.5)

    ax2 = axes[1]
    pitch.draw(ax=ax2)
    ax2.invert_yaxis()
    
    # Split event locations by team control to compute cell-level dominance
    x1 = df.loc[df["team_ball_control"]==1, "ball_x_m"].values
    y1 = df.loc[df["team_ball_control"]==1, "ball_y_m"].values
    x2 = df.loc[df["team_ball_control"]==2, "ball_x_m"].values
    y2 = df.loc[df["team_ball_control"]==2, "ball_y_m"].values
    
    # Bin into a coarse grid to emphasize territory rather than micro-variations
    bin_x, bin_y = 6, 4
    stats1 = pitch.bin_statistic(x1, y1, statistic='count', bins=(bin_x, bin_y))
    stats2 = pitch.bin_statistic(x2, y2, statistic='count', bins=(bin_x, bin_y))
    
    # Normalize to a dominance score in [-1, 1] while handling empty cells safely
    count1 = stats1['statistic']
    count2 = stats2['statistic']
    total = count1 + count2
    
    with np.errstate(divide='ignore', invalid='ignore'):
        dominance = (count1 - count2) / total
    dominance[np.isnan(dominance)] = 0 

    # Use a diverging palette to encode team dominance symmetrically
    cmap_div = LinearSegmentedColormap.from_list("dom_cmap", ['#dc143c', '#444444', '#00bfff'], N=100)
    stats1['statistic'] = dominance
    pitch.heatmap(stats1, ax=ax2, cmap=cmap_div, vmin=-0.8, vmax=0.8, edgecolors='#22312b', alpha=0.9)
    
    ax2.set_title("Territorial Control (Blue vs Red)", color='white', fontsize=14, fontweight='bold')

    ax3 = axes[2]
    ax3.set_facecolor('#22312b')
    
    # Compute defensive/middle/attacking third shares with team-direction-aware thresholds
    def get_zone_distribution(team_id):
        sub = df[df["team_ball_control"] == team_id]
        if len(sub) == 0: return [0, 0, 0]
        
        x = sub["ball_x_m"]
        limit1 = pitch_length / 3
        limit2 = 2 * pitch_length / 3
        
        if team_id == 1:
            defs = len(x[x > limit2])
            mids = len(x[(x >= limit1) & (x <= limit2)])
            atts = len(x[x < limit1])
        else:
            defs = len(x[x < limit1])
            mids = len(x[(x >= limit1) & (x <= limit2)])
            atts = len(x[x > limit2])
            
        total_z = defs + mids + atts
        if total_z == 0: return [0, 0, 0]
        return [defs/total_z*100, mids/total_z*100, atts/total_z*100]

    # Compute third distribution for both teams for side-by-side comparison
    dist1 = get_zone_distribution(1)
    dist2 = get_zone_distribution(2)
    
    # Plot a grouped bar chart to compare territorial presence by third
    labels = ["Defensive", "Midfield", "Attacking"]
    x = np.arange(len(labels))
    width = 0.35
    
    rects1 = ax3.bar(x - width/2, dist1, width, label=team1_name, color='#00bfff', alpha=0.9)
    rects2 = ax3.bar(x + width/2, dist2, width, label=team2_name, color='#dc143c', alpha=0.9)
    
    ax3.bar_label(rects1, fmt='%.0f%%', padding=3, color='white', fontweight='bold')
    ax3.bar_label(rects2, fmt='%.0f%%', padding=3, color='white', fontweight='bold')
    
    # Style the panel for dark-mode reporting and ensure a consistent 0–100% scale
    ax3.set_title("Territory Share (Team Perspective)", color='white', fontsize=14, fontweight='bold')
    ax3.set_ylabel("Share (%)", color='white')
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, color='white', fontsize=11)
    ax3.set_ylim(0, 100) 
    
    ax3.spines['bottom'].set_color('white')
    ax3.spines['left'].set_color('white')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.tick_params(axis='y', colors='white')
    
    # Add a legend suitable for slides while preserving dark-mode readability
    legend = ax3.legend(facecolor='#22312b', edgecolor='white')
    for text in legend.get_texts(): text.set_color("white")

    plt.tight_layout()
    return fig