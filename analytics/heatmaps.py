import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple


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
    bins: Tuple[int, int] = (40, 24),
    ax: plt.Axes | None = None,
    cmap: str = "hot",
    alpha: float = 0.6,
):
    """
    Zeichnet den Ball-Heatmap-Layer auf dein Pitch-Bild.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    # Hintergrund
    ax.imshow(pitch_img, extent=[0, pitch_length, 0, pitch_width], aspect="equal")

    H, xedges, yedges = compute_ball_heatmap(df, pitch_length, pitch_width, bins=bins)

    ax.imshow(
        H.T,
        extent=[0, pitch_length, 0, pitch_width],
        origin="lower",
        cmap=cmap,
        alpha=alpha,
        aspect="equal",
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
    bins: Tuple[int, int] = (40, 24),
):
    """
    Zeichnet 2 Heatmaps nebeneinander:
    - Team 1 Ballpositionen nur bei Ballkontrolle von Team 1
    - Team 2 entsprechend
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

        ax.imshow(
            pitch_img,
            extent=[0, pitch_length, 0, pitch_width],
            aspect="equal",
        )

        if len(x) > 0:
            H, xedges, yedges = np.histogram2d(
                x,
                y,
                bins=bins,
                range=[[0, pitch_length], [0, pitch_width]],
            )
            ax.imshow(
                H.T,
                extent=[0, pitch_length, 0, pitch_width],
                origin="lower",
                cmap="hot",
                alpha=0.6,
                aspect="equal",
            )

        ax.set_xlim(0, pitch_length)
        ax.set_ylim(0, pitch_width)
        ax.set_xlabel("Länge (m)")
        ax.set_title(f"Team {team_id} – Ball-Heatmap")

    axes[0].set_ylabel("Breite (m)")
    fig.suptitle("Ball-Heatmaps nach Ballkontrolle-Team", fontsize=14)
    fig.tight_layout()
    return fig, axes