import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage import gaussian_filter1d

# Dark Mode Setup
BG_COLOR = '#22312b'
TEXT_COLOR = 'white'

def compute_rolling_possession(df: pd.DataFrame, window_sec: float) -> pd.DataFrame:
    """
    Berechnet rollenden Ballbesitz.
    """
    # Sortieren zur Sicherheit
    df = df.sort_values("time_sec")
    
    time = df["time_sec"].to_numpy()
    ctrl = df["team_ball_control"].to_numpy()

    # dt schätzen
    dt = np.nanmedian(np.diff(time))
    if not np.isfinite(dt) or dt <= 0:
        dt = 1.0/30.0 # Fallback
        
    window_frames = max(1, int(window_sec / dt))

    is_t1 = (ctrl == 1).astype(float)
    is_t2 = (ctrl == 2).astype(float)

    # Rolling Mean
    roll_t1 = pd.Series(is_t1).rolling(window_frames, min_periods=1).mean()
    roll_t2 = pd.Series(is_t2).rolling(window_frames, min_periods=1).mean()

    # Glättung für schöne Kurven (Sigma relativ zum Fenster)
    sigma = window_frames / 4
    roll_t1_smooth = gaussian_filter1d(roll_t1, sigma)
    roll_t2_smooth = gaussian_filter1d(roll_t2, sigma)

    out = pd.DataFrame({
        "time_min": df["time_sec"] / 60.0,
        "roll_t1": roll_t1_smooth * 100,
        "roll_t2": roll_t2_smooth * 100,
    })
    return out


def plot_rolling_possession(df: pd.DataFrame, team1_name="Team 1", team2_name="Team 2", window_sec=30.0, ax=None):
    """
    Erstellt einen 'Match Momentum' Graphen im Dark Mode.
    Unterstützt jetzt 'ax' für Subplots.
    """
    stats = compute_rolling_possession(df, window_sec=window_sec)
    
    if len(stats) == 0:
        return

    # Setup Plot: Entweder existierende Achse (ax) oder neues Bild
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 3)) # Standardmäßig flacher
        fig.set_facecolor(BG_COLOR)
    else:
        fig = ax.get_figure() # Referenz holen

    ax.set_facecolor(BG_COLOR)

    # Daten
    x = stats["time_min"]
    y1 = stats["roll_t1"]
    y2 = stats["roll_t2"]
    
    # Dominanz berechnen (Positiv = Team 1, Negativ = Team 2)
    dominance = y1 - y2 

    # 1. Team 1 Dominanz (Blau füllen)
    ax.fill_between(x, 0, dominance, where=(dominance > 0), 
                    interpolate=True, color='#00bfff', alpha=0.6, label=team1_name)
    
    # 2. Team 2 Dominanz (Rot füllen)
    ax.fill_between(x, 0, dominance, where=(dominance <= 0), 
                    interpolate=True, color='#dc143c', alpha=0.6, label=team2_name)

    # 3. Nulllinie
    ax.axhline(0, color='#c7d5cc', linewidth=1, alpha=0.5, linestyle='--')

    # Styling
    ax.set_title("Match Momentum (Dominanz)", color=TEXT_COLOR, fontsize=14, fontweight='bold', pad=10)
    ax.set_xlabel("Spielzeit (Minuten)", color=TEXT_COLOR)
    
    # Achsen-Farben
    ax.spines['bottom'].set_color(TEXT_COLOR)
    ax.spines['left'].set_color(TEXT_COLOR)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(colors=TEXT_COLOR)
    
    # Y-Achse beschriften oder ausblenden
    ax.set_yticks([]) # Keine Zahlen, da Index abstrakt ist
    
    # Legende (sauberer als Text an festen Koordinaten)
    legend = ax.legend(loc='upper right', facecolor=BG_COLOR, edgecolor='None', fontsize=9)
    for text in legend.get_texts():
        text.set_color(TEXT_COLOR)

    # Nur layouten, wenn wir das Bild selbst erstellt haben
    if ax is None:
        plt.tight_layout()
        plt.show()
    
    return fig