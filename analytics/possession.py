import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage import gaussian_filter1d

# Centralized dark-mode styling for consistent report visuals
BG_COLOR = '#22312b'
TEXT_COLOR = 'white'


# Compute a smoothed rolling possession share (%) per team over a configurable time window
def compute_rolling_possession(df: pd.DataFrame, window_sec: float) -> pd.DataFrame:
    # Ensure correct temporal ordering before rolling aggregation
    df = df.sort_values("time_sec")
    
    time = df["time_sec"].to_numpy()
    ctrl = df["team_ball_control"].to_numpy()

    # Estimate frame interval from data to convert seconds into rolling frames
    dt = np.nanmedian(np.diff(time))
    if not np.isfinite(dt) or dt <= 0:
        dt = 1.0/30.0 # Fallback
        
    window_frames = max(1, int(window_sec / dt))

    # Encode possession as binary indicators for rolling mean computation
    is_t1 = (ctrl == 1).astype(float)
    is_t2 = (ctrl == 2).astype(float)

    # Compute rolling possession shares per team within the window
    roll_t1 = pd.Series(is_t1).rolling(window_frames, min_periods=1).mean()
    roll_t2 = pd.Series(is_t2).rolling(window_frames, min_periods=1).mean()

    # Apply Gaussian smoothing to reduce volatility and improve interpretability
    sigma = window_frames / 4
    roll_t1_smooth = gaussian_filter1d(roll_t1, sigma)
    roll_t2_smooth = gaussian_filter1d(roll_t2, sigma)

    # Return time-aligned rolling percentages for downstream plotting
    out = pd.DataFrame({
        "time_min": df["time_sec"] / 60.0,
        "roll_t1": roll_t1_smooth * 100,
        "roll_t2": roll_t2_smooth * 100,
    })
    return out


# Plot a dark-mode match momentum chart based on rolling possession dominance (Team 1 minus Team 2)
def plot_rolling_possession(df: pd.DataFrame, team1_name="Team 1", team2_name="Team 2", window_sec=30.0, ax=None):
    # Compute rolling possession statistics as the basis for the momentum visualization
    stats = compute_rolling_possession(df, window_sec=window_sec)
    
    # Exit early when no valid statistics are available
    if len(stats) == 0:
        return

    # Create a new figure or draw into an existing axis for dashboard layouts
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 3)) # Standardmäßig flacher
        fig.set_facecolor(BG_COLOR)
    else:
        fig = ax.get_figure() # Referenz holen

    ax.set_facecolor(BG_COLOR)

    # Extract time series for both teams in percent units
    x = stats["time_min"]
    y1 = stats["roll_t1"]
    y2 = stats["roll_t2"]
    
    # Compute dominance signal (positive: Team 1, negative: Team 2)
    dominance = y1 - y2 

    # Fill positive dominance intervals to highlight Team 1 momentum phases
    ax.fill_between(x, 0, dominance, where=(dominance > 0), 
                    interpolate=True, color='#00bfff', alpha=0.6, label=team1_name)
    
    # Fill negative dominance intervals to highlight Team 2 momentum phases
    ax.fill_between(x, 0, dominance, where=(dominance <= 0), 
                    interpolate=True, color='#dc143c', alpha=0.6, label=team2_name)

    # Add a zero baseline to separate momentum swings clearly
    ax.axhline(0, color='#c7d5cc', linewidth=1, alpha=0.5, linestyle='--')

    # Apply report-ready labels and dark-mode styling
    ax.set_title("Match Momentum (Dominance)", color=TEXT_COLOR, fontsize=14, fontweight='bold', pad=10)
    ax.set_xlabel("Match Time (Minutes)", color=TEXT_COLOR)
    
    # Harmonize axis spines and ticks for dark backgrounds
    ax.spines['bottom'].set_color(TEXT_COLOR)
    ax.spines['left'].set_color(TEXT_COLOR)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(colors=TEXT_COLOR)
    
    # Hide y-axis ticks since the dominance index is relative, not absolute
    ax.set_yticks([])
    
    # Use a compact legend for clear team attribution in dashboards and exports
    legend = ax.legend(loc='upper right', facecolor=BG_COLOR, edgecolor='None', fontsize=9)
    for text in legend.get_texts():
        text.set_color(TEXT_COLOR)

    # Finalize layout only when the figure is created within this function
    if ax is None:
        plt.tight_layout()
        plt.show()
    
    return fig