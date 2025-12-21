import numpy as np
from sklearn.cluster import KMeans
import cv2
from collections import defaultdict


class TeamAssigner:
    # Estimate team and referee colors and infer per-player team labels using robust color heuristics
    def __init__(self):
        # Store canonical jersey colors per team for distance-based inference
        self.team_colors = {}       # mean jersey colors for each team: {1: (B,G,R), 2: (B,G,R)}
        # Store an optional referee color prototype to support debugging and future filtering
        self.referee_color = None   # estimated average referee color in BGR

    def get_clustering_model(self, image):
        # Use a lightweight KMeans baseline to split foreground/background dominant colors
        image_2d = image.reshape(-1, 3)
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1)
        kmeans.fit(image_2d)
        return kmeans

    def get_player_color(self, frame, bbox):
        # Clamp bbox to frame boundaries to avoid empty crops and OpenCV errors
        x1, y1, x2, y2 = map(int, bbox)
        h_frame, w_frame = frame.shape[:2]

        x1 = max(0, min(w_frame - 1, x1))
        x2 = max(0, min(w_frame,     x2))
        y1 = max(0, min(h_frame - 1, y1))
        y2 = max(0, min(h_frame,     y2))

        # Crop the player region and fall back to a neutral color when detection is invalid
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return np.array([170, 170, 170], dtype=np.float32)

        # Focus on torso area to reduce contamination from shorts, socks, and nearby players
        h, w = crop.shape[:2]
        y_top    = int(h * 0.25)
        y_bottom = int(h * 0.75)
        x_left   = int(w * 0.20)
        x_right  = int(w * 0.80)
        torso = crop[y_top:y_bottom, x_left:x_right]

        # Fall back to full crop when torso window is empty (e.g., very small boxes)
        if torso.size == 0:
            torso = crop
            h, w = torso.shape[:2]
        else:
            h, w = torso.shape[:2]

        # Filter grass pixels in HSV space to reduce background influence on jersey color estimation
        torso_hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)

        H = torso_hsv[:, :, 0]
        S = torso_hsv[:, :, 1]
        V = torso_hsv[:, :, 2]

        grass_mask = (
            (H >= 30) & (H <= 90) &
            (S >= 40) & (V >= 40)
        )

        # Treat remaining pixels as player evidence, but fall back when too few survive masking
        player_mask = ~grass_mask
        player_pixels = torso[player_mask]
        if player_pixels.shape[0] < 10:
            pixels = torso.reshape(-1, 3)
        else:
            pixels = player_pixels.reshape(-1, 3)

        if len(pixels) < 10:
            return np.array([170, 170, 170], dtype=np.float32)

        # Cluster jersey candidates and pick the most central cluster to reduce edge contamination
        k = 3
        km = KMeans(n_clusters=k, init="k-means++", n_init=5, random_state=0)
        km.fit(pixels)

        # Project labels back to torso grid to compute which cluster dominates the center region
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

        # Use the cluster closest to the torso center as the jersey color proxy
        best_cluster = int(np.argmin(centrality))
        player_color = centers[best_cluster]

        return player_color

    def assign_referee_color(self, frames, tracks, sample_frames=50):
        # Aggregate referee bbox colors across frames to build a robust color reference
        colors = []
        F = min(sample_frames, len(frames))
        for f in range(F):
            for _, r in tracks["referees"][f].items():
                colors.append(self.get_player_color(frames[f], r["bbox"]))

        # Leave referee_color unset when no referee detections are available
        if not colors:
            self.referee_color = None
            return

        # Use median color to reduce sensitivity to outliers and mixed crops
        X = np.asarray(colors, dtype=np.float32)
        median_color = np.median(X, axis=0)
        self.referee_color = median_color.astype(np.float32)

    def assign_team_color(self, frames, tracks, sample_frames=10):
        # Prefer a geometry-based left/right split in early frames to avoid color-only ambiguity
        num_frames = len(frames)
        F = min(sample_frames, num_frames)

        # Collect player center-x samples to infer the natural team separation line
        cx_samples = []

        for f in range(F):
            frame = frames[f]
            h, w = frame.shape[:2]

            for _, p in tracks["players"][f].items():
                x1, y1, x2, y2 = map(int, p["bbox"])
                cx = 0.5 * (x1 + x2)
                if 0 <= cx < w:
                    cx_samples.append(cx)

        # Use 1D clustering on x-positions to estimate the team split when enough evidence exists
        MIN_CX_SAMPLES = 20
        if len(cx_samples) >= MIN_CX_SAMPLES:
            X_cx = np.asarray(cx_samples, dtype=np.float32).reshape(-1, 1)

            km_split = KMeans(n_clusters=2, n_init=10, random_state=0).fit(X_cx)
            centers = np.sort(km_split.cluster_centers_.flatten())

            x_left_center, x_right_center = centers[0], centers[1]
            split_x = 0.5 * (x_left_center + x_right_center)
        else:
            # Fall back to image midpoint when early tracking is sparse or unreliable
            if len(frames) == 0:
                self.team_colors = {1: (0, 255, 255), 2: (255, 0, 0)}
                return
            h, w = frames[0].shape[:2]
            split_x = w * 0.5

        # Collect jersey colors per side to compute stable team prototypes
        colors_left = []   # left group  -> Team 2
        colors_right = []  # right group -> Team 1

        for f in range(F):
            frame = frames[f]
            for _, p in tracks["players"][f].items():
                bbox = p["bbox"]
                x1, y1, x2, y2 = map(int, bbox)
                cx = 0.5 * (x1 + x2)

                color = self.get_player_color(frame, bbox)

                # Assign by side to enforce consistent team labeling from kickoff orientation
                if cx <= split_x:
                    colors_left.append(color)
                else:
                    colors_right.append(color)

        # Use per-side medians when enough samples exist to avoid mixing team distributions
        MIN_COLORS_PER_SIDE = 10
        if len(colors_left) >= MIN_COLORS_PER_SIDE and len(colors_right) >= MIN_COLORS_PER_SIDE:
            X_left = np.asarray(colors_left, dtype=np.float32)
            X_right = np.asarray(colors_right, dtype=np.float32)

            c_left = np.median(X_left, axis=0)
            c_right = np.median(X_right, axis=0)

            # Persist canonical colors with a fixed convention: Team 1 = right, Team 2 = left
            self.team_colors[1] = tuple(map(int, c_right))
            self.team_colors[2] = tuple(map(int, c_left))
            return

        # Fall back to color-only clustering when geometry-based sampling is insufficient
        colors = []
        F = min(sample_frames, len(frames))
        for f in range(F):
            frame = frames[f]
            for _, p in tracks["players"][f].items():
                colors.append(self.get_player_color(frame, p["bbox"]))

        if len(colors) < 3:
            self.team_colors = {1: (0, 255, 255), 2: (255, 0, 0)}
            return

        X = np.asarray(colors, dtype=np.float32)

        # Use k=3 to allow an outlier cluster (often referees or mixed crops) to be ignored
        k3 = KMeans(n_clusters=3, n_init=10, random_state=0).fit(X)
        labels3 = k3.labels_
        counts = np.bincount(labels3, minlength=3)

        top2 = counts.argsort()[-2:][::-1]
        cands = k3.cluster_centers_[top2]

        # Revert to k=2 when the two largest clusters are not clearly separable
        if np.linalg.norm(cands[0] - cands[1]) < 25:
            k2 = KMeans(n_clusters=2, n_init=10, random_state=0).fit(X)
            cands = k2.cluster_centers_

        c1, c2 = cands
        self.team_colors[1] = tuple(map(int, c1))
        self.team_colors[2] = tuple(map(int, c2))

    def get_player_team(self, frame, player_bbox, player_id=None):
        # Delegate to the current bbox-only inference to keep older pipeline code working
        return self.infer_team_for_bbox(frame, player_bbox)

    def infer_team_for_bbox(self, frame, bbox):
        # Use LAB color space for more perceptual distance behavior than raw BGR
        color = self.get_player_color(frame, bbox).astype(np.float32)
        color_lab = self._bgr_to_lab(color)
        c1_lab = self._bgr_to_lab(self.team_colors[1])
        c2_lab = self._bgr_to_lab(self.team_colors[2])

        d1 = np.linalg.norm(color_lab - c1_lab)
        d2 = np.linalg.norm(color_lab - c2_lab)

        return 1 if d1 < d2 else 2
    
    def save_color_debug(self, out_path="output_video_match/color_debug.png"):
        # Skip export when team prototypes are not yet available
        if 1 not in self.team_colors or 2 not in self.team_colors:
            return

        # Create a simple canvas to visualize reference colors consistently
        h = 160
        w = 600
        img = np.full((h, w, 3), 230, np.uint8)

        # Render labeled color blocks to support quick visual sanity checks
        def draw_block(x, color_bgr, label):
            x1, y1 = x, 30
            x2, y2 = x + 150, 120
            cv2.rectangle(img, (x1, y1), (x2, y2), tuple(int(c) for c in color_bgr), -1)
            cv2.rectangle(img, (x1, y1), (x2, y2), (40, 40, 40), 1)
            cv2.putText(img, label, (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2, cv2.LINE_AA)

        # Visualize canonical colors for both teams
        c1 = self.team_colors[1]
        c2 = self.team_colors[2]
        draw_block(20,  c1, f"Team 1 {c1}")
        draw_block(220, c2, f"Team 2 {c2}")

        # Include referee prototype when available to validate separation from team colors
        if getattr(self, "referee_color", None) is not None:
            cref = tuple(int(x) for x in np.asarray(self.referee_color).tolist())
            draw_block(420, cref, f"Ref {cref}")
        else:
            cv2.putText(img, "kein Ref-Farbcluster", (400, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 2, cv2.LINE_AA)

        cv2.imwrite(out_path, img)

    def _bgr_to_lab(self, color_bgr):
        # Normalize into OpenCV image shape and reuse cvtColor for consistent conversion
        col = np.asarray(color_bgr, dtype=np.uint8).reshape(1, 1, 3)
        lab = cv2.cvtColor(col, cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)
        return lab