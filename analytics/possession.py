import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def compute_rolling_possession(df: pd.DataFrame, window_sec: float) -> pd.DataFrame:
    """
    Berechnet rollenden Ballbesitz (Team 1 & Team 2) über ein Zeitfenster in Sekunden.
    Gibt einen DataFrame mit Spalten 'time_min', 'roll_t1', 'roll_t2' zurück.
    """
    time = df["time_sec"].to_numpy()
    ctrl = df["team_ball_control"].to_numpy()

    # dt schätzen
    dt = np.nanmedian(np.diff(time))
    if not np.isfinite(dt) or dt <= 0:
        raise RuntimeError("Konnte dt / FPS nicht sinnvoll bestimmen.")
    window_frames = max(1, int(window_sec / dt))

    is_t1 = (ctrl == 1).astype(float)
    is_t2 = (ctrl == 2).astype(float)

    roll_t1 = pd.Series(is_t1).rolling(window_frames, min_periods=1).mean()
    roll_t2 = pd.Series(is_t2).rolling(window_frames, min_periods=1).mean()

    out = pd.DataFrame(
        {
            "time_min": df["time_sec"] / 60.0,   # <<< Minuten statt Sekunden
            "roll_t1": roll_t1,
            "roll_t2": roll_t2,
        }
    )
    return out


def plot_rolling_possession(df: pd.DataFrame, window_sec: float = 10.0):
    """
    Plottet rollenden Ballbesitz (Team 1 & Team 2) über die Zeit in MINUTEN.
    """
    stats = compute_rolling_possession(df, window_sec=window_sec)

    plt.figure(figsize=(10, 4))
    plt.plot(stats["time_min"], stats["roll_t1"] * 100.0, label="Team 1")
    plt.plot(stats["time_min"], stats["roll_t2"] * 100.0, label="Team 2")

    plt.xlabel("Zeit (min)")  # <<< jetzt Minuten
    plt.ylabel(f"Ballbesitz in den letzten {window_sec:.0f}s (%)")
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title("Rolling Ball Possession (in Minuten)")

    plt.tight_layout()
    return plt.gca()