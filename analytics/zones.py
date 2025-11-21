# project/Computer-Vision-FCH/analytics/zones.py

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List


def _zone(x: float, pitch_length: float) -> str:
    if x < pitch_length / 3:
        return "Defensiv"
    if x < 2 * pitch_length / 3:
        return "Mittelfeld"
    return "Offensiv"


def compute_zone_percentages(df, pitch_length: float) -> Dict[str, List[float]]:
    """
    Berechnet Zonen-Anteile (in %) für:
      - Gesamt
      - Team 1 (bei Ballkontrolle)
      - Team 2
    Gibt ein Dict mit Keys 'zones', 'total', 'team1', 'team2'.
    """
    mask = df["ball_x_m"].notna()
    df_z = df.loc[mask].copy()
    df_z["zone"] = df_z["ball_x_m"].apply(lambda x: _zone(x, pitch_length))

    zones = ["Defensiv", "Mittelfeld", "Offensiv"]

    def zone_pct(sub):
        counts = sub["zone"].value_counts(normalize=True)
        return [counts.get(z, 0.0) * 100.0 for z in zones]

    total_pct = zone_pct(df_z)
    t1_pct = zone_pct(df_z[df_z["team_ball_control"] == 1])
    t2_pct = zone_pct(df_z[df_z["team_ball_control"] == 2])

    return {
        "zones": zones,
        "total": total_pct,
        "team1": t1_pct,
        "team2": t2_pct,
    }


def plot_zone_summary(
    df,
    pitch_img,
    pitch_length: float,
    pitch_width: float,
):
    """
    Erzeugt 3 nebeneinanderstehende Plots:
      1) Tabelle der Zonenanteile
      2) Balkendiagramm (Gesamt vs Team 1 vs Team 2)
      3) Pitch mit Zonen und Prozentangaben
    """
    stats = compute_zone_percentages(df, pitch_length)
    zones = stats["zones"]
    total_pct = stats["total"]
    t1_pct = stats["team1"]
    t2_pct = stats["team2"]

    fig, axes = plt.subplots(1, 3, figsize=(22, 6))

    # 1) Tabelle
    ax = axes[0]
    ax.axis("off")

    table_data = [
        ["Zone", "Gesamt %", "Team 1 %", "Team 2 %"],
        [zones[0], f"{total_pct[0]:.1f}", f"{t1_pct[0]:.1f}", f"{t2_pct[0]:.1f}"],
        [zones[1], f"{total_pct[1]:.1f}", f"{t1_pct[1]:.1f}", f"{t2_pct[1]:.1f}"],
        [zones[2], f"{total_pct[2]:.1f}", f"{t1_pct[2]:.1f}", f"{t2_pct[2]:.1f}"],
    ]

    table = ax.table(cellText=table_data, loc="center", cellLoc="center")
    table.scale(1.2, 1.8)
    ax.set_title("Zonen – Tabellenübersicht", fontsize=14)

    # 2) Balkendiagramm
    ax = axes[1]
    x = np.arange(len(zones))
    width = 0.25

    ax.bar(x - width, total_pct, width, label="Gesamt")
    ax.bar(x, t1_pct, width, label="Team 1")
    ax.bar(x + width, t2_pct, width, label="Team 2")

    ax.set_xticks(x, zones)
    ax.set_ylabel("Anteil Ballzeit (%)")
    ax.set_title("Zonenverteilung (Balkendiagramm)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    # 3) Pitch mit Zonenwerten
    ax = axes[2]
    L = pitch_length

    ax.imshow(pitch_img, extent=[0, L, 0, pitch_width], aspect="equal")

    ax.axvline(L / 3, linestyle="--", color="white")
    ax.axvline(2 * L / 3, linestyle="--", color="white")

    for i, name in enumerate(zones):
        x_center = (i + 0.5) * (L / 3)

        ax.text(
            x_center,
            pitch_width * 0.65,
            f"{name}",
            ha="center",
            va="center",
            fontsize=11,
            color="white",
            bbox=dict(facecolor="black", alpha=0.5, boxstyle="round,pad=0.3"),
        )

        ax.text(
            x_center,
            pitch_width * 0.45,
            f"Gesamt: {total_pct[i]:.1f}%",
            ha="center",
            va="center",
            fontsize=10,
            color="white",
            bbox=dict(facecolor="black", alpha=0.4, boxstyle="round,pad=0.2"),
        )

        ax.text(
            x_center,
            pitch_width * 0.30,
            f"T1: {t1_pct[i]:.1f}%",
            ha="center",
            va="center",
            fontsize=10,
            color="white",
            bbox=dict(facecolor="blue", alpha=0.4, boxstyle="round,pad=0.2"),
        )

        ax.text(
            x_center,
            pitch_width * 0.17,
            f"T2: {t2_pct[i]:.1f}%",
            ha="center",
            va="center",
            fontsize=10,
            color="white",
            bbox=dict(facecolor="red", alpha=0.4, boxstyle="round,pad=0.2"),
        )

    ax.set_xlim(0, L)
    ax.set_ylim(0, pitch_width)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Zonenverteilung auf dem Feld")

    fig.tight_layout()
    return fig, axes