import numpy as np
from sklearn.cluster import KMeans
import cv2

class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}

    def get_clustering_model(self, image):
        image_2d = image.reshape(-1, 3)
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1)
        kmeans.fit(image_2d)
        return kmeans

    def get_player_color(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return np.array([170,170,170], dtype=np.float32)  # neutral fallback
        # mittleres Trikotband ist robuster
        h = crop.shape[0]
        band = crop[int(h*0.3):int(h*0.7), :]
        if band.size == 0:
            band = crop
        km = self.get_clustering_model(band)
        labels = km.labels_.reshape(band.shape[0], band.shape[1])
        # Ecken = Hintergrund → wähle den anderen Cluster
        corner = [labels[0,0], labels[0,-1], labels[-1,0], labels[-1,-1]]
        non_player = max(set(corner), key=corner.count)
        player_cluster = 1 - non_player
        return km.cluster_centers_[player_cluster].astype(np.float32)  # BGR floats

    def assign_team_color(self, frames, tracks, sample_frames=10):
        # 1) Farben über mehrere Frames sammeln (nur Spieler)
        colors = []
        F = min(sample_frames, len(frames))
        for f in range(F):
            for _, p in tracks["players"][f].items():
                colors.append(self.get_player_color(frames[f], p["bbox"]))
        if len(colors) < 3:
            # Notfall – neutrale Defaults
            self.team_colors = {1:(0,255,255), 2:(255,0,0)}
            return

        X = np.asarray(colors, dtype=np.float32)

        # 2) Erst K=3 clustern, um Schiri/Outlier als kleinen Cluster abzuspalten
        k3 = KMeans(n_clusters=3, n_init=10, random_state=0).fit(X)
        labels3 = k3.labels_
        counts = np.bincount(labels3, minlength=3)
        # Indizes der zwei größten Cluster
        top2 = counts.argsort()[-2:][::-1]
        cands = k3.cluster_centers_[top2]

        # 3) Sicherheitscheck: sind die zwei Zentren „weit genug“ auseinander?
        if np.linalg.norm(cands[0]-cands[1]) < 25:  # Distanz in BGR
            # Fallback auf K=2
            k2 = KMeans(n_clusters=2, n_init=10, random_state=0).fit(X)
            cands = k2.cluster_centers_

        c1, c2 = cands
        self.team_colors[1] = tuple(map(int, c1))
        self.team_colors[2] = tuple(map(int, c2))

    def get_player_team(self, frame, player_bbox, player_id):
        # Konsistenz über die Zeit
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]
        color = self.get_player_color(frame, player_bbox)
        d1 = np.linalg.norm(color - np.asarray(self.team_colors[1], dtype=np.float32))
        d2 = np.linalg.norm(color - np.asarray(self.team_colors[2], dtype=np.float32))
        team = 1 if d1 < d2 else 2
        self.player_team_dict[player_id] = team
        return team