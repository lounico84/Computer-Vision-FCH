import numpy as np
from sklearn.cluster import KMeans
import cv2
from collections import defaultdict

class TeamAssigner:
    def __init__(self):
        self.team_colors = {}       # mean jersey colors for each team: {1: (B,G,R), 2: (B,G,R)}
        self.referee_color = None   # estimated average referee color in BGR

    # Run k-means clustering on the image pixels to find 2 dominat color clusters
    def get_clustering_model(self, image):
        image_2d = image.reshape(-1, 3)
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1)
        kmeans.fit(image_2d)
        return kmeans

    # Estimate the dominant jersey color of a player or referee
    def get_player_color(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        h_frame, w_frame = frame.shape[:2]

        # BBox clampen
        x1 = max(0, min(w_frame - 1, x1))
        x2 = max(0, min(w_frame,     x2))
        y1 = max(0, min(h_frame - 1, y1))
        y2 = max(0, min(h_frame,     y2))

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return np.array([170, 170, 170], dtype=np.float32)

        # Torso wie bisher
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

        # --- NEU: Rasen herausfiltern über HSV ---
        torso_hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)

        # Typischer Grünbereich: H ~ [30, 90], ausreichend S/V
        H = torso_hsv[:, :, 0]
        S = torso_hsv[:, :, 1]
        V = torso_hsv[:, :, 2]

        grass_mask = (
            (H >= 30) & (H <= 90) &
            (S >= 40) & (V >= 40)
        )

        # Spieler-Pixel = alles, was nicht offensichtlich Rasen ist
        player_mask = ~grass_mask

        # Falls nach Filter zu wenig Pixel -> fallback auf ursprüngliche Logik
        player_pixels = torso[player_mask]
        if player_pixels.shape[0] < 10:
            pixels = torso.reshape(-1, 3)
        else:
            pixels = player_pixels.reshape(-1, 3)

        if len(pixels) < 10:
            return np.array([170, 170, 170], dtype=np.float32)

        k = 3
        km = KMeans(n_clusters=k, init="k-means++", n_init=5, random_state=0)
        km.fit(pixels)

        # Wir brauchen Labels in Bildform nur, wenn wir zentrale Cluster berechnen.
        # Hier einfacher: cluster-Centroid nehmen, der am nächsten zur Bildmitte liegt.
        # Also gleichen Ansatz wie vorher, dafür benötigen wir labels.
        labels_full = km.predict(torso.reshape(-1, 3)).reshape(h, w)
        centers = km.cluster_centers_.astype(np.float32)

        ys, xs = np.indices((h, w))
        cx, cy = w / 2.0, h / 2.0

        centrality = []
        for c in range(k):
            mask = (labels_full == c)
            if not np.any(mask):
                centrality.append(np.inf)
                continue
            xs_c = xs[mask].astype(np.float32)
            ys_c = ys[mask].astype(np.float32)
            dist2 = (xs_c - cx) ** 2 + (ys_c - cy) ** 2
            centrality.append(np.mean(dist2))

        best_cluster = int(np.argmin(centrality))
        player_color = centers[best_cluster]

        return player_color

    # Estimate an average referee color from bbox in tracks['referees']
    def assign_referee_color(self, frames, tracks, sample_frames=50):

        colors = []
        F = min(sample_frames, len(frames))
        for f in range(F):
            for _, r in tracks["referees"][f].items():
                colors.append(self.get_player_color(frames[f], r["bbox"]))

        if not colors:
            # No referee detections, cannot estimate a referee color
            self.referee_color = None
            return

        X = np.asarray(colors, dtype=np.float32)
        median_color = np.median(X, axis=0)
        self.referee_color = median_color.astype(np.float32)

    # Estimate the main jersey colors for team 1 and 2
    # Estimate the main jersey colors for team 1 and 2
    def assign_team_color(self, frames, tracks, sample_frames=10):
        """
        Verbesserte Version:
        1) Nutzt in den ersten Frames die X-Positionen der Spieler, um eine sinnvolle
           Trennlinie zwischen zwei Gruppen (linkes vs. rechtes Team) zu finden.
           -> Linke Gruppe  = Team 2
           -> Rechte Gruppe = Team 1
        2) Wenn das nicht klappt (zu wenig Samples), Fallback auf das alte K-Means-Verfahren.
        """
        num_frames = len(frames)
        F = min(sample_frames, num_frames)

        # Erstmal nur alle Spieler-X-Mitten sammeln, um die Trennlinie zu bestimmen
        cx_samples = []

        for f in range(F):
            frame = frames[f]
            h, w = frame.shape[:2]

            for _, p in tracks["players"][f].items():
                x1, y1, x2, y2 = map(int, p["bbox"])
                cx = 0.5 * (x1 + x2)
                # Nur BBoxen im Bildbereich
                if 0 <= cx < w:
                    cx_samples.append(cx)

        # Wenn wir nicht genug Samples haben -> direkt Fallback
        MIN_CX_SAMPLES = 20
        if len(cx_samples) >= MIN_CX_SAMPLES:
            X_cx = np.asarray(cx_samples, dtype=np.float32).reshape(-1, 1)

            # 1D-K-Means über die X-Koordinaten der Spieler
            km_split = KMeans(n_clusters=2, n_init=10, random_state=0).fit(X_cx)
            centers = np.sort(km_split.cluster_centers_.flatten())

            # linkes und rechtes Clusterzentrum
            x_left_center, x_right_center = centers[0], centers[1]
            # Trennlinie zwischen den beiden Gruppen
            split_x = 0.5 * (x_left_center + x_right_center)
        else:
            # zu wenig Daten für sinnvolle Trennlinie -> Fallback auf Bildmitte
            # (kommt selten vor, z.B. wenn Tracking im ersten Frame noch leer ist)
            # Hinweis: das ist nur ein Notbehelf, Hauptfall ist der km_split oben.
            if len(frames) == 0:
                self.team_colors = {1: (0, 255, 255), 2: (255, 0, 0)}
                return
            h, w = frames[0].shape[:2]
            split_x = w * 0.5

        # Jetzt aus den ersten Frames pro Seite die Trikotfarben sammeln
        colors_left = []   # linke Gruppe  -> Team 2
        colors_right = []  # rechte Gruppe -> Team 1

        for f in range(F):
            frame = frames[f]
            for _, p in tracks["players"][f].items():
                bbox = p["bbox"]
                x1, y1, x2, y2 = map(int, bbox)
                cx = 0.5 * (x1 + x2)

                color = self.get_player_color(frame, bbox)

                # Strikte Zuordnung:
                # alles links/auf der Trennlinie -> Team 2 (left),
                # alles rechts davon           -> Team 1 (right)
                if cx <= split_x:
                    colors_left.append(color)
                else:
                    colors_right.append(color)

        # Wenn wir genug Samples pro Seite haben, nutzen wir sie direkt
        MIN_COLORS_PER_SIDE = 10
        if len(colors_left) >= MIN_COLORS_PER_SIDE and len(colors_right) >= MIN_COLORS_PER_SIDE:
            X_left = np.asarray(colors_left, dtype=np.float32)
            X_right = np.asarray(colors_right, dtype=np.float32)

            # Median je Seite: robust gegenüber Ausreißern
            c_left = np.median(X_left, axis=0)
            c_right = np.median(X_right, axis=0)

            # WICHTIG:
            # Team 1 = rechte Gruppe
            # Team 2 = linke Gruppe
            self.team_colors[1] = tuple(map(int, c_right))  # rechts
            self.team_colors[2] = tuple(map(int, c_left))   # links
            return

        # -----------------------------------------
        # Fallback: altes K-Means-Verfahren (wie bisher)
        # -----------------------------------------
        colors = []
        F = min(sample_frames, len(frames))
        for f in range(F):
            frame = frames[f]
            for _, p in tracks["players"][f].items():
                colors.append(self.get_player_color(frame, p["bbox"]))

        if len(colors) < 3:
            # Fallback, simple default colors if not enough samples
            self.team_colors = {1: (0, 255, 255), 2: (255, 0, 0)}
            return

        X = np.asarray(colors, dtype=np.float32)

        # Cluster with K=3 to split off referee/outlier as a small cluster
        k3 = KMeans(n_clusters=3, n_init=10, random_state=0).fit(X)
        labels3 = k3.labels_
        counts = np.bincount(labels3, minlength=3)
        # Indices of the two largest clusters (likely team 1 and 2)
        top2 = counts.argsort()[-2:][::-1]
        cands = k3.cluster_centers_[top2]

        # Check if the clusters are sufficiently different
        if np.linalg.norm(cands[0] - cands[1]) < 25:  # distance in BGR space
            # If too close fallback to K=2 clustering
            k2 = KMeans(n_clusters=2, n_init=10, random_state=0).fit(X)
            cands = k2.cluster_centers_

        c1, c2 = cands
        self.team_colors[1] = tuple(map(int, c1))
        self.team_colors[2] = tuple(map(int, c2))

    # Return a temporary team label for the player and collect statistics
    ## How many times a plaxer looks like team 1 or 2
    ## How often a player looks similar to the referee color
    def get_player_team(self, frame, player_bbox, player_id=None):
        return self.infer_team_for_bbox(frame, player_bbox)

    # Infer the team 1 or 2 for a bbox using only color
    def infer_team_for_bbox(self, frame, bbox):
        color = self.get_player_color(frame, bbox).astype(np.float32)
        color_lab = self._bgr_to_lab(color)
        c1_lab = self._bgr_to_lab(self.team_colors[1])
        c2_lab = self._bgr_to_lab(self.team_colors[2])

        d1 = np.linalg.norm(color_lab - c1_lab)
        d2 = np.linalg.norm(color_lab - c2_lab)

        return 1 if d1 < d2 else 2
    
    # Save debug image showing three key colors of team 1, 2 and referee
    def save_color_debug(self, out_path="output_video_match/color_debug.png"):

        # If team colors are not set yet, there is nothing to visualize
        if 1 not in self.team_colors or 2 not in self.team_colors:
            return

        # Create a blank image as a canvas
        h = 160
        w = 600
        img = np.full((h, w, 3), 230, np.uint8)

        # Helper function to draw a colored rectangle with a text label
        def draw_block(x, color_bgr, label):
            x1, y1 = x, 30
            x2, y2 = x + 150, 120
            cv2.rectangle(img, (x1, y1), (x2, y2), tuple(int(c) for c in color_bgr), -1)
            cv2.rectangle(img, (x1, y1), (x2, y2), (40, 40, 40), 1)
            cv2.putText(img, label, (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2, cv2.LINE_AA)

        # Team 1 and 2 color blocks
        c1 = self.team_colors[1]
        c2 = self.team_colors[2]
        draw_block(20,  c1, f"Team 1 {c1}")
        draw_block(220, c2, f"Team 2 {c2}")

        # Referee color block
        if getattr(self, "referee_color", None) is not None:
            cref = tuple(int(x) for x in np.asarray(self.referee_color).tolist())
            draw_block(420, cref, f"Ref {cref}")
        else:
            # If no referee color cluster was found, display a text
            cv2.putText(img, "kein Ref-Farbcluster", (400, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 2, cv2.LINE_AA)

        cv2.imwrite(out_path, img)

    def _bgr_to_lab(self, color_bgr):
        col = np.asarray(color_bgr, dtype=np.uint8).reshape(1, 1, 3)
        lab = cv2.cvtColor(col, cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)
        return lab