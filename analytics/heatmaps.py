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
    """
    Erstellt eine 'Smooth Heatmap' im Dark Mode für ein spezifisches Team.
    Unterstützt jetzt Subplots via 'ax'.
    """
    # Daten filtern
    mask = (df["team_ball_control"] == team_id) & df["ball_x_m"].notna() & df["ball_y_m"].notna()
    x = df.loc[mask, "ball_x_m"].to_numpy()
    y = df.loc[mask, "ball_y_m"].to_numpy()

    if len(x) == 0:
        return

    # Pitch Setup
    pitch = Pitch(
        pitch_type='custom',
        pitch_length=pitch_length,
        pitch_width=pitch_width,
        line_color='#c7d5cc',
        pitch_color='#22312b',
        linewidth=2,
    )

    # Entweder neue Figure oder existierende Achse nutzen
    if ax is None:
        fig, ax = pitch.draw(figsize=(10, 6))
        fig.set_facecolor('#22312b')
    else:
        pitch.draw(ax=ax)

    ax.invert_yaxis()

    # Statistik berechnen (Binning)
    bin_statistic = pitch.bin_statistic(x, y, statistic='count', bins=(25, 15))
    heatmap_data = bin_statistic['statistic']

    # Glätten (Gaussian Filter für den "Pro"-Look)
    heatmap_smooth = gaussian_filter(heatmap_data, sigma=1.5)

    # Colormap erstellen
    cmap = LinearSegmentedColormap.from_list("team_cmap", [color_start, color_end], N=100)

    # Maskieren (sehr niedrige Werte ausblenden)
    heatmap_smooth[heatmap_smooth < 0.5] = np.nan
    
    # Werte zurückschreiben und plotten
    bin_statistic['statistic'] = heatmap_smooth
    pitch.heatmap(bin_statistic, ax=ax, cmap=cmap, edgecolors=None, alpha=0.9)

    # Titel (Größe angepasst für Subplots)
    ax.set_title(f"{team_name}", color='white', fontsize=16, fontweight='bold', pad=10)


def plot_team_ball_heatmaps_on_pitch(df, pitch_img, pitch_length, pitch_width):
    """
    Legacy Funktion: Erzeugt 2 Subplots (Links Team 1, Rechts Team 2).
    Kann genutzt werden, wenn man die Heatmaps schnell isoliert sehen will.
    """
    team_cfg = getattr(s, "team_names", None)
    t1_name = team_cfg.team1_name if team_cfg else "Team 1"
    t2_name = team_cfg.team2_name if team_cfg else "Team 2"

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.set_facecolor('#22312b')

    # Team 1
    plot_pro_heatmap(
        df, team_id=1, 
        pitch_length=pitch_length, pitch_width=pitch_width, 
        ax=axes[0], team_name=t1_name,
        color_start="#22312b", color_end="#00bfff"
    )

    # Team 2
    plot_pro_heatmap(
        df, team_id=2, 
        pitch_length=pitch_length, pitch_width=pitch_width, 
        ax=axes[1], team_name=t2_name,
        color_start="#22312b", color_end="#ff4d4d" # Rot
    )

    plt.tight_layout()
    return fig, axes

# Kompatibilitäts-Funktionen (falls alte Pipeline sie aufruft)
def compute_ball_heatmap(*args, **kwargs): 
    return None, None, None

def plot_ball_heatmap_on_pitch(df, pitch_img, pitch_length, pitch_width, bins=None, ax=None, cmap="jet"):
    """
    Erstellt eine Gesamt-Heatmap (beide Teams) im Pro-Look.
    """
    pitch = Pitch(
        pitch_type='custom', pitch_length=pitch_length, pitch_width=pitch_width,
        line_color='#c7d5cc', pitch_color='#22312b', linewidth=2
    )
    
    if ax is None:
        fig, ax = pitch.draw(figsize=(10, 6))
        fig.set_facecolor('#22312b')
    else:
        pitch.draw(ax=ax)
        
    ax.invert_yaxis()

    mask = df["ball_x_m"].notna() & df["ball_y_m"].notna()
    x = df.loc[mask, "ball_x_m"].to_numpy()
    y = df.loc[mask, "ball_y_m"].to_numpy()

    if len(x) > 0:
        bin_statistic = pitch.bin_statistic(x, y, statistic='count', bins=(30, 20))
        heatmap_smooth = gaussian_filter(bin_statistic['statistic'], sigma=1.5)
        
        # Goldene Heatmap für Gesamtübersicht
        cmap_gold = LinearSegmentedColormap.from_list("gold_cmap", ["#22312b", "#ffd700"], N=100)
        heatmap_smooth[heatmap_smooth < 1] = np.nan
        
        bin_statistic['statistic'] = heatmap_smooth
        pitch.heatmap(bin_statistic, ax=ax, cmap=cmap_gold, alpha=0.9)

    ax.set_title("Gesamt Heatmap (Ball)", color='white', fontsize=16, fontweight='bold', pad=10)
    return ax