import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from config import Settings

s = Settings()
analytics_cfg = s.analytics

def plot_tactical_formation(df, pitch_img, length, width, team1_name, team2_name, n_clusters=10):
    """
    Ignoriert Tracking-IDs und nutzt K-Means Clustering, um die 
    wahrscheinlichsten Positionen der Spieler zu finden.
    n_clusters = 10 (wir ignorieren meist den Torwart oder Auswechselspieler bei Ballbesitz)
    """
    # Wir filtern nur Frames, in denen der Ball kontrolliert wurde
    # und ignorieren Ausreißer/Fehler (z.B. Ball im Aus) durch eine einfache Rand-Logik
    margin = 2.0
    valid_touches = df[
        (df["ball_x_m"] > margin) & (df["ball_x_m"] < length - margin) &
        (df["ball_y_m"] > margin) & (df["ball_y_m"] < width - margin)
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.invert_yaxis()
    ax.imshow(pitch_img, extent=[0, length, width, 0], alpha=0.8)
    
    # Helper für das Plotten eines Teams
    def plot_team_clusters(team_id, color, label_name, marker_style):
        # 1. Alle Koordinaten dieses Teams holen
        points = valid_touches[valid_touches["team_ball_control"] == team_id][["ball_x_m", "ball_y_m"]]
        
        # Safety Check: Haben wir genug Datenpunkte für K-Means?
        if len(points) < n_clusters:
            print(f"Zu wenige Daten für {label_name} ({len(points)} Punkte).")
            return

        # 2. K-Means berechnet die Schwerpunkte (Centroids)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(points)
        centers = kmeans.cluster_centers_

        # 3. Plotten der Zentren
        ax.scatter(centers[:, 0], centers[:, 1], 
                   c=color, s=350, edgecolors="white", linewidth=2, 
                   label=label_name, zorder=3, alpha=0.9)
        
        # Optional: Nummerieren der Positionen (nur zur Unterscheidung)
        # Wir sortieren sie nach X (Defensiv -> Offensiv), damit die Nummern logisch wirken
        sorted_indices = np.argsort(centers[:, 0])
        for i, idx in enumerate(sorted_indices):
            cx, cy = centers[idx]
            # Wir schreiben keine ID, sondern eine Pseudo-Positionsnummer (1=Torwartnähe, 10=Sturm)
            # Oder wir lassen es leer, weil es keine echten Rückennummern sind.
            # Hier: Leer lassen oder Symbol.
            ax.text(cx, cy, "x", color="white", ha="center", va="center", fontweight="bold")

    # Team 1 (Blau)
    plot_team_clusters(1, "blue", team1_name, "o")
    
    # Team 2 (Rot)
    plot_team_clusters(2, "red", team2_name, "o")

    ax.set_title(f"Taktische Grundordnung (Ballaktionen geclustert)\n{team1_name} vs {team2_name}")
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.01), ncol=2)
    ax.axis("off") # Achsen ausblenden für cleanen Look
    
    plt.tight_layout()
    plt.show()
