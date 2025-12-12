import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from mplsoccer import Pitch
from scipy.spatial import ConvexHull          # <--- NEU für die Hüllkurve
from matplotlib.patches import Polygon

from config import Settings
s = Settings()

def plot_tactical_formation(df, pitch_img, length, width, team1_name, team2_name, n_clusters=10):
    """
    Erstellt eine Taktik-Tafel mit Durchschnittspositionen UND Raumaufteilung (Convex Hull).
    """
    
    # 1. Datenvorbereitung
    margin = 2.0
    valid_touches = df[
        (df["ball_x_m"] > margin) & (df["ball_x_m"] < length - margin) &
        (df["ball_y_m"] > margin) & (df["ball_y_m"] < width - margin)
    ]

    # 2. SETUP Pitch
    pitch = Pitch(
        pitch_type='custom', pitch_length=length, pitch_width=width,
        line_color='#c7d5cc', pitch_color='#22312b', linewidth=2,
    )

    fig, ax = pitch.draw(figsize=(10, 6))
    fig.set_facecolor('#22312b')
    ax.invert_yaxis() # Oben ist Oben

    # --- HELPER FUNKTION ---
    def plot_team_clusters(team_id, color, label_name, text_color='white'):
        points = valid_touches[valid_touches["team_ball_control"] == team_id][["ball_x_m", "ball_y_m"]]
        
        if len(points) < n_clusters:
            return

        # K-Means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(points)
        centers = kmeans.cluster_centers_

        # Sortierung
        sorted_indices = np.argsort(centers[:, 0])
        if team_id == 2: sorted_indices = sorted_indices[::-1]

        # --- NEU: CONVEX HULL (Das "Gummiband" um das Team) ---
        if len(centers) > 2: # Braucht mind. 3 Punkte für eine Fläche
            hull = ConvexHull(centers)
            # Eckpunkte des Polygons holen
            hull_points = centers[hull.vertices]
            
            # Polygon zeichnen (Transparente Fläche)
            poly = Polygon(hull_points, closed=True, 
                           facecolor=color, alpha=0.1, # Sehr zart gefüllt
                           edgecolor=color, linewidth=2, linestyle='--') # Gestrichelter Rand
            ax.add_patch(poly)

        # --- NEU: TEAM SCHWERPUNKT (Centroid) ---
        centroid_x = np.mean(centers[:, 0])
        centroid_y = np.mean(centers[:, 1])
        pitch.scatter(centroid_x, centroid_y, marker='X', s=100, color=color, 
                      edgecolors='white', ax=ax, zorder=1, alpha=0.6, label='Schwerpunkt')

        # PLOTTING DER SPIELER
        for i, idx in enumerate(sorted_indices):
            cx, cy = centers[idx]
            
            # Einflussbereich (Glow)
            pitch.scatter(cx, cy, s=1200, color=color, alpha=0.15, edgecolors='none', ax=ax, zorder=2)
            # Spieler (Kern)
            pitch.scatter(cx, cy, s=350, color=color, edgecolors='white', linewidth=2, ax=ax, zorder=3)
            # Nummer
            ax.text(cx, cy, str(i+1), color=text_color, ha="center", va="center", fontsize=9, fontweight="bold", zorder=4)

    # Teams Plotten
    plot_team_clusters(1, "#00bfff", team1_name, text_color="black") # Cyan
    plot_team_clusters(2, "#dc143c", team2_name, text_color="white") # Rot

    # Titel
    fig.text(0.5, 0.95, f"Taktische Formation & Raumaufteilung", 
             color='white', fontsize=18, fontweight='bold', ha='center', va='center')
    
    # Untertitel mit Erklärung
    fig.text(0.5, 0.90, f"{team1_name} vs {team2_name} | Gestrichelt: Abgedeckter Raum", 
             color='#c7d5cc', fontsize=10, ha='center', va='center')

    plt.show()