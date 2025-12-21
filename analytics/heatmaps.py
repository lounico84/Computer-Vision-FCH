import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter
from mplsoccer import Pitch

from config import Settings
s = Settings()

def plot_pro_heatmap(
    df, 
    team_id, 
    pitch_length, 
    pitch_width, 
    ax=None, 
    team_name="", 
    color_start="#22312b", 
    color_end="white"
):
    # Select valid team-controlled ball locations as heatmap input events
    mask = (df["team_ball_control"] == team_id) & df["ball_x_m"].notna() & df["ball_y_m"].notna()
    x = df.loc[mask, "ball_x_m"].to_numpy()
    y = df.loc[mask, "ball_y_m"].to_numpy()

    # Exit early when there is no data to visualize for this team
    if len(x) == 0:
        return

    # Build a pitch canvas using the same dimensions as the calibrated model space
    pitch = Pitch(
        pitch_type='custom',
        pitch_length=pitch_length,
        pitch_width=pitch_width,
        line_color='#c7d5cc',
        pitch_color='#22312b',
        linewidth=2,
    )

    # Render either a standalone figure or draw onto a provided subplot axis
    if ax is None:
        fig, ax = pitch.draw(figsize=(10, 6))
        fig.set_facecolor('#22312b')
    else:
        pitch.draw(ax=ax)

    # Standardize orientation so heatmaps align with the tactical view
    ax.invert_yaxis()

    # Aggregate events into a 2D grid to quantify spatial frequency
    bin_statistic = pitch.bin_statistic(x, y, statistic='count', bins=(25, 15))
    heatmap_data = bin_statistic['statistic']

    # Smooth the binned counts to reduce discretization artifacts and improve readability
    heatmap_smooth = gaussian_filter(heatmap_data, sigma=1.5)

    # Create a team-specific gradient colormap for consistent visual identity
    cmap = LinearSegmentedColormap.from_list("team_cmap", [color_start, color_end], N=100)

    # Suppress low-density areas to emphasize meaningful occupation zones
    heatmap_smooth[heatmap_smooth < 0.5] = np.nan
    
    # Plot the smoothed grid back onto the pitch using the configured palette
    bin_statistic['statistic'] = heatmap_smooth
    pitch.heatmap(bin_statistic, ax=ax, cmap=cmap, edgecolors=None, alpha=0.9)

    # Add a concise subplot-ready title for reporting and dashboards
    ax.set_title(f"{team_name}", color='white', fontsize=16, fontweight='bold', pad=10)


def plot_team_ball_heatmaps_on_pitch(df, pitch_img, pitch_length, pitch_width):
    # Resolve team names from configuration with safe defaults
    team_cfg = getattr(s, "team_names", None)
    t1_name = team_cfg.team1_name if team_cfg else "Team 1"
    t2_name = team_cfg.team2_name if team_cfg else "Team 2"

    # Create a side-by-side layout for direct team comparison
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.set_facecolor('#22312b')

    # Render Team 1 heatmap using a cyan gradient for clear differentiation
    plot_pro_heatmap(
        df, team_id=1, 
        pitch_length=pitch_length, pitch_width=pitch_width, 
        ax=axes[0], team_name=t1_name,
        color_start="#22312b", color_end="#00bfff"
    )

    # Render Team 2 heatmap using a red gradient for clear differentiation
    plot_pro_heatmap(
        df, team_id=2, 
        pitch_length=pitch_length, pitch_width=pitch_width, 
        ax=axes[1], team_name=t2_name,
        color_start="#22312b", color_end="#ff4d4d" # Rot
    )

    # Optimize spacing for export-ready figures
    plt.tight_layout()
    return fig, axes

# Preserve API compatibility for legacy pipeline calls without producing output
def compute_ball_heatmap(*args, **kwargs): 
    return None, None, None

def plot_ball_heatmap_on_pitch(df, pitch_img, pitch_length, pitch_width, bins=None, ax=None, cmap="jet"):
    # Build a pitch canvas consistent with other analytics visualizations
    pitch = Pitch(
        pitch_type='custom', pitch_length=pitch_length, pitch_width=pitch_width,
        line_color='#c7d5cc', pitch_color='#22312b', linewidth=2
    )
    
    # Support both standalone figure creation and subplot embedding
    if ax is None:
        fig, ax = pitch.draw(figsize=(10, 6))
        fig.set_facecolor('#22312b')
    else:
        pitch.draw(ax=ax)
        
    # Standardize orientation so aggregation matches all other pitch plots
    ax.invert_yaxis()

    # Use all valid ball locations to produce an overall occupancy map
    mask = df["ball_x_m"].notna() & df["ball_y_m"].notna()
    x = df.loc[mask, "ball_x_m"].to_numpy()
    y = df.loc[mask, "ball_y_m"].to_numpy()

    # Only compute and render the heatmap when observations exist
    if len(x) > 0:
        bin_statistic = pitch.bin_statistic(x, y, statistic='count', bins=(30, 20))
        heatmap_smooth = gaussian_filter(bin_statistic['statistic'], sigma=1.5)
        
        # Apply a high-contrast gold palette for a single combined overview
        cmap_gold = LinearSegmentedColormap.from_list("gold_cmap", ["#22312b", "#ffd700"], N=100)
        heatmap_smooth[heatmap_smooth < 1] = np.nan
        
        # Render the smoothed aggregate intensity layer on the pitch
        bin_statistic['statistic'] = heatmap_smooth
        pitch.heatmap(bin_statistic, ax=ax, cmap=cmap_gold, alpha=0.9)

    # Provide an explicit chart title for report-ready export
    ax.set_title("Gesamt Heatmap (Ball)", color='white', fontsize=16, fontweight='bold', pad=10)
    return ax