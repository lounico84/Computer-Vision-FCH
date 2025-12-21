import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from mplsoccer import Pitch
from scipy.spatial import ConvexHull
from matplotlib.patches import Polygon

from config import Settings
s = Settings()

def plot_tactical_formation(df, pitch_img, length, width, team1_name, team2_name, n_clusters=10, ax=None):
    
    # Filter to in-bounds ball events to reduce noise near pitch edges
    margin = 2.0
    valid_touches = df[
        (df["ball_x_m"] > margin) & (df["ball_x_m"] < length - margin) &
        (df["ball_y_m"] > margin) & (df["ball_y_m"] < width - margin)
    ]

    # Configure a custom pitch canvas aligned with the calibrated field dimensions
    pitch = Pitch(
        pitch_type='custom', pitch_length=length, pitch_width=width,
        line_color='#c7d5cc', pitch_color='#22312b', linewidth=2,
    )

    # Support both standalone plots and embedding into existing subplot axes
    if ax is None:
        fig, ax = pitch.draw(figsize=(10, 6))
        fig.set_facecolor('#22312b')
    else:
        pitch.draw(ax=ax)

    # Align coordinate system so "upfield" matches the visual orientation
    ax.invert_yaxis()

    # Encapsulate team-specific clustering and plotting for reuse across both sides
    def plot_team_clusters(team_id, color, label_name, text_color='white'):
        # Extract team-controlled ball locations as proxy for team shape and occupation
        points = valid_touches[valid_touches["team_ball_control"] == team_id][["ball_x_m", "ball_y_m"]]
        
        # Skip clustering if sample size is too small to produce stable centers
        if len(points) < n_clusters:
            return

        # Cluster touch locations to derive representative positional "nodes"
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(points)
        centers = kmeans.cluster_centers_

        # Order centers left-to-right (and mirror for the opposite team) for consistent numbering
        sorted_indices = np.argsort(centers[:, 0])
        if team_id == 2: sorted_indices = sorted_indices[::-1]

        # Compute convex hull over centers to approximate the team's covered area
        if len(centers) > 2:
            hull = ConvexHull(centers)
            hull_points = centers[hull.vertices]
            
            # Visualize occupied space as a semi-transparent boundary overlay
            poly = Polygon(hull_points, closed=True, 
                           facecolor=color, alpha=0.1, 
                           edgecolor=color, linewidth=2, linestyle='--')
            ax.add_patch(poly)

        # Mark overall center of mass to summarize team compactness/field tilt
        centroid_x = np.mean(centers[:, 0])
        centroid_y = np.mean(centers[:, 1])
        pitch.scatter(centroid_x, centroid_y, marker='X', s=100, color=color, 
                      edgecolors='white', ax=ax, zorder=1, alpha=0.6, label='Schwerpunkt')

        # Render each cluster center as a numbered node with a light influence halo
        for i, idx in enumerate(sorted_indices):
            cx, cy = centers[idx]
            
            pitch.scatter(cx, cy, s=1200, color=color, alpha=0.15, edgecolors='none', ax=ax, zorder=2)
            pitch.scatter(cx, cy, s=350, color=color, edgecolors='white', linewidth=2, ax=ax, zorder=3)
            ax.text(cx, cy, str(i+1), color=text_color, ha="center", va="center", fontsize=9, fontweight="bold", zorder=4)

    # Plot both teams with distinct colors to support side-by-side tactical interpretation
    plot_team_clusters(1, "#00bfff", team1_name, text_color="black")  # Cyan
    plot_team_clusters(2, "#dc143c", team2_name, text_color="white")  # Red

    # Add narrative context depending on whether this is a standalone figure or a subplot
    if ax is None:
        fig.text(0.5, 0.95, f"Taktische Formation & Raumaufteilung", 
                 color='white', fontsize=18, fontweight='bold', ha='center', va='center')
        fig.text(0.5, 0.90, f"{team1_name} vs {team2_name} | Gestrichelt: Abgedeckter Raum", 
                 color='#c7d5cc', fontsize=10, ha='center', va='center')
        plt.show()
        return fig
    else:
        ax.set_title("Formation & Kompaktheit", color='white', fontsize=14, fontweight='bold')