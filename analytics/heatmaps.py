import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
import cv2

def compute_ball_heatmap(
    df,
    pitch_length: float,
    pitch_width: float,
    bins: Tuple[int, int] = (40, 24),
):
    """
    Berechnet eine 2D-Heatmap des Balles (in Meterkoordinaten).
    Gibt (H, xedges, yedges) zurück.
    """
    mask = df["ball_x_m"].notna() & df["ball_y_m"].notna()
    x = df.loc[mask, "ball_x_m"].to_numpy()
    y = df.loc[mask, "ball_y_m"].to_numpy()

    H, xedges, yedges = np.histogram2d(
        x,
        y,
        bins=bins,
        range=[[0, pitch_length], [0, pitch_width]],
    )
    return H, xedges, yedges


def plot_ball_heatmap_on_pitch(
    df,
    pitch_img,
    pitch_length: float,
    pitch_width: float,
    bins: Tuple[int, int] = (120, 72),
    ax: plt.Axes | None = None,
    cmap: str = "jet",
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    # Hintergrund: sehr leicht sichtbar
    ax.imshow(
        pitch_img,
        extent=[0, pitch_length, 0, pitch_width],
        aspect="equal",
        alpha=0.35,     # <--- reduziert Feld-Dominanz
    )

    # Ballpositionen
    mask = df["ball_x_m"].notna() & df["ball_y_m"].notna()
    x = df.loc[mask, "ball_x_m"].to_numpy()
    y = df.loc[mask, "ball_y_m"].to_numpy()

    # Histogramm mit hoher Auflösung
    H, xedges, yedges = np.histogram2d(
        x, y,
        bins=bins,
        range=[[0, pitch_length], [0, pitch_width]],
    )

    H = H.astype(np.float32)

    # Gaussian Blur = weich/rund
    H_blur = cv2.GaussianBlur(H, (0, 0), sigmaX=3.5, sigmaY=3.5)

    # Perzentil-Normalisierung (viel besser als max-Norm)
    hi = np.nanpercentile(H_blur, 98)
    if hi > 0:
        H_norm = np.clip(H_blur / hi, 0, 1)
    else:
        H_norm = H_blur

    # Kleine Werte entfernen
    H_norm[H_norm < 0.05] = 0.0

    # Weiche Anzeige
    ax.imshow(
        H_norm.T,
        extent=[0, pitch_length, 0, pitch_width],
        origin="lower",
        cmap=cmap,
        interpolation="bilinear",
        alpha=0.85,      # <--- Heatmap deutlich sichtbarer
        vmin=0,
        vmax=1,
    )

    ax.set_xlim(0, pitch_length)
    ax.set_ylim(0, pitch_width)
    ax.set_xlabel("Länge (m)")
    ax.set_ylabel("Breite (m)")
    ax.set_title("Ball-Heatmap")

    return ax


def plot_team_ball_heatmaps_on_pitch(
    df,
    pitch_img,
    pitch_length: float,
    pitch_width: float,
    bins: Tuple[int, int] = (90, 54),
):
    """
    Glatte Heatmaps für Team 1 und Team 2 nebeneinander.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for team_id, ax in zip([1, 2], axes):
        mask = (
            df["ball_x_m"].notna()
            & df["ball_y_m"].notna()
            & (df["team_ball_control"] == team_id)
        )
        x = df.loc[mask, "ball_x_m"].to_numpy()
        y = df.loc[mask, "ball_y_m"].to_numpy()

        # Pitch
        ax.imshow(
            pitch_img,
            extent=[0, pitch_length, 0, pitch_width],
            aspect="equal",
            alpha=1.0,
        )

        if len(x) > 0:
            H, xedges, yedges = np.histogram2d(
                x, y,
                bins=bins,
                range=[[0, pitch_length], [0, pitch_width]],
            )

            H = H.astype(np.float32)
            H_blur = cv2.GaussianBlur(H, ksize=(0, 0), sigmaX=2.0, sigmaY=2.0)

            max_val = np.max(H_blur)
            if max_val > 0:
                H_norm = H_blur / max_val
            else:
                H_norm = H_blur

            H_norm[H_norm < 0.02] = 0.0

            ax.imshow(
                H_norm.T,
                extent=[0, pitch_length, 0, pitch_width],
                origin="lower",
                cmap="jet",
                alpha=0.65,
                aspect="equal",
                vmin=0,
                vmax=1,
                interpolation="bilinear",
            )

        ax.set_xlim(0, pitch_length)
        ax.set_ylim(0, pitch_width)
        ax.set_xlabel("Länge (m)")
        ax.set_title(f"Team {team_id} – Ball-Heatmap")

    axes[0].set_ylabel("Breite (m)")
    fig.suptitle("Ball-Heatmaps nach Ballkontrolle-Team", fontsize=14)
    fig.tight_layout()
    return fig, axes