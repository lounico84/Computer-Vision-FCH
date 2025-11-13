import numpy as np
from sklearn.cluster import KMeans
import cv2

class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}
        self.referee_color = None
        self.ref_vote_counts = {}
        self.player_obs_counts = {}
        self.team_vote_counts = {} 

    def get_clustering_model(self, image):
        image_2d = image.reshape(-1, 3)
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1)
        kmeans.fit(image_2d)
        return kmeans

    def get_player_color(self, frame, bbox):
        """
        Bestimmt die Trikotfarbe eines Spielers/Schiris:
        - Torso-Zone ausschneiden
        - KMeans(K=3) über Farben
        - Den Cluster wählen, dessen Pixel räumlich am nächsten
        zum Bildzentrum liegen (meist Körper/Torso).
        """
        x1, y1, x2, y2 = map(int, bbox)
        h_frame, w_frame = frame.shape[:2]

        # BBox auf Bildgrenzen clampen
        x1 = max(0, min(w_frame - 1, x1))
        x2 = max(0, min(w_frame,     x2))
        y1 = max(0, min(h_frame - 1, y1))
        y2 = max(0, min(h_frame,     y2))

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return np.array([170, 170, 170], dtype=np.float32)  # Fallback

        # Torso-Zone: mittleres Band
        h, w = crop.shape[:2]
        y_top    = int(h * 0.25)
        y_bottom = int(h * 0.75)
        x_left   = int(w * 0.20)
        x_right  = int(w * 0.80)
        torso = crop[y_top:y_bottom, x_left:x_right]
        if torso.size == 0:
            torso = crop
            h, w = torso.shape[:2]
        else:
            h, w = torso.shape[:2]

        # Daten für KMeans vorbereiten
        pixels = torso.reshape(-1, 3)  # BGR
        if len(pixels) < 10:
            return np.array([170, 170, 170], dtype=np.float32)

        k = 3
        km = KMeans(n_clusters=k, init="k-means++", n_init=5, random_state=0)
        km.fit(pixels)
        labels = km.labels_.reshape(h, w)          # Cluster-Label pro Pixel
        centers = km.cluster_centers_.astype(np.float32)  # (3,3) BGR

        # Koordinatenraster
        ys, xs = np.indices((h, w))
        cx, cy = w / 2.0, h / 2.0

        # Für jeden Cluster: durchschnittliche Distanz seiner Pixel zum Bildzentrum
        centrality = []
        for c in range(k):
            mask = (labels == c)
            if not np.any(mask):
                centrality.append(np.inf)
                continue
            xs_c = xs[mask].astype(np.float32)
            ys_c = ys[mask].astype(np.float32)
            dist2 = (xs_c - cx) ** 2 + (ys_c - cy) ** 2
            centrality.append(np.mean(dist2))

        # Cluster mit Pixeln, die am ehesten in der Mitte liegen
        best_cluster = int(np.argmin(centrality))
        player_color = centers[best_cluster]

        return player_color

    def assign_referee_color(self, frames, tracks, sample_frames=50):
        """
        Bestimmt eine mittlere Referee-Farbe aus BBoxen in tracks['referees'].
        Nutzt die gleiche get_player_color-Logik, aber nur auf Referees/Linienrichter.
        """
        colors = []
        F = min(sample_frames, len(frames))
        for f in range(F):
            for _, r in tracks["referees"][f].items():
                colors.append(self.get_player_color(frames[f], r["bbox"]))

        if not colors:
            self.referee_color = None
            return

        X = np.asarray(colors, dtype=np.float32)
        median_color = np.median(X, axis=0)
        self.referee_color = median_color.astype(np.float32)

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
        """
        Gibt ein temporäres Team-Label (1 oder 2) zurück und sammelt
        gleichzeitig Statistiken, wie oft ein Track Ref-ähnlich ist.
        Die endgültige Entscheidung (Ref oder nicht) passiert später.
        """
        color = self.get_player_color(frame, player_bbox).astype(np.float32)

        # Beobachtungen zählen
        self.player_obs_counts[player_id] = self.player_obs_counts.get(player_id, 0) + 1

        # Distanz zu Teamfarben
        c1 = np.asarray(self.team_colors[1], dtype=np.float32)
        c2 = np.asarray(self.team_colors[2], dtype=np.float32)
        d1 = np.linalg.norm(color - c1)
        d2 = np.linalg.norm(color - c2)

        team = 1 if d1 < d2 else 2

        # Ref-Ähnlichkeit nur zählen, noch NICHT umlabeln
        if self.referee_color is not None:
            cref = np.asarray(self.referee_color, dtype=np.float32)
            dref = np.linalg.norm(color - cref)

            # Helligkeit prüfen (nur dunkle Kandidaten)
            col_uint8 = color.astype(np.uint8).reshape(1, 1, 3)
            h, s, v = cv2.cvtColor(col_uint8, cv2.COLOR_BGR2HSV)[0, 0]
            if dref < 0.6 * min(d1, d2):
                prev = self.ref_vote_counts.get(player_id, 0)
                self.ref_vote_counts[player_id] = prev + 1
        
        # NEU: Team-Votes sammeln
        votes = self.team_vote_counts.get(player_id)
        if votes is None:
            votes = {1: 0, 2: 0}
            self.team_vote_counts[player_id] = votes
        votes[team] += 1

        return team
        
    def save_color_debug(self, out_path="output_video_match/color_debug.png"):
        """
        Speichert ein kleines Bild mit den drei wichtigsten Farben:
        - Team 1
        - Team 2
        - Referee (falls vorhanden)
        """
        # Falls Teamfarben noch nicht gesetzt sind: nichts tun
        if 1 not in self.team_colors or 2 not in self.team_colors:
            return

        # Bild anlegen
        h = 160
        w = 600
        img = np.full((h, w, 3), 230, np.uint8)

        # Helper zum Zeichnen eines Blocks + Text
        def draw_block(x, color_bgr, label):
            x1, y1 = x, 30
            x2, y2 = x + 150, 120
            cv2.rectangle(img, (x1, y1), (x2, y2), tuple(int(c) for c in color_bgr), -1)
            cv2.rectangle(img, (x1, y1), (x2, y2), (40, 40, 40), 1)
            cv2.putText(img, label, (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2, cv2.LINE_AA)

        # Team 1 und Team 2
        c1 = self.team_colors[1]
        c2 = self.team_colors[2]
        draw_block(20,  c1, f"Team 1 {c1}")
        draw_block(220, c2, f"Team 2 {c2}")

        # Referee (falls vorhanden)
        if getattr(self, "referee_color", None) is not None:
            cref = tuple(int(x) for x in np.asarray(self.referee_color).tolist())
            draw_block(420, cref, f"Ref {cref}")
        else:
            cv2.putText(img, "kein Ref-Farbcluster", (400, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 2, cv2.LINE_AA)

        cv2.imwrite(out_path, img)