import numpy as np
from sklearn.cluster import KMeans
import cv2

class TeamAssigner:
    def __init__(self):
        self.team_colors = {}       # mean jersey colors for each team: {1: (B,G,R), 2: (B,G,R)}
        self.player_team_dict = {}  # final mapping: player_id to team_id 1 or 2
        self.referee_color = None   # estimated average referee color in BGR
        self.ref_vote_counts = {}   # how many times each plaxer was classified as "referee-like"
        self.player_obs_counts = {} # how many times a player was seen
        self.team_vote_counts = {}  # how many votes for team 1 and team 2 {player_id: {1: count_team1, 2: count_team2}}

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

        # Clamp bbox coordinates to valid image range
        x1 = max(0, min(w_frame - 1, x1))
        x2 = max(0, min(w_frame,     x2))
        y1 = max(0, min(h_frame - 1, y1))
        y2 = max(0, min(h_frame,     y2))

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return np.array([170, 170, 170], dtype=np.float32)  # fallback neutral gray if crop is invalid

        # Define torso region as a central band within the bbox
        h, w = crop.shape[:2]
        y_top    = int(h * 0.25)
        y_bottom = int(h * 0.75)
        x_left   = int(w * 0.20)
        x_right  = int(w * 0.80)
        torso = crop[y_top:y_bottom, x_left:x_right]

        # If torso region is empty, fallback to full crop
        if torso.size == 0:
            torso = crop
            h, w = torso.shape[:2]
        else:
            h, w = torso.shape[:2]

        # Prepare data for k-means (flatten to 2D: pixels x 3 channels)
        pixels = torso.reshape(-1, 3)  # BGR
        if len(pixels) < 10:
            # Not enough pixels to cluster, return neutral color
            return np.array([170, 170, 170], dtype=np.float32)

        k = 3
        km = KMeans(n_clusters=k, init="k-means++", n_init=5, random_state=0)
        km.fit(pixels)
        labels = km.labels_.reshape(h, w)          # cluster label per pixel
        centers = km.cluster_centers_.astype(np.float32)  # (3,3) BGR

        # Coordinate grid for centrality calculation
        ys, xs = np.indices((h, w))
        cx, cy = w / 2.0, h / 2.0

        # For each cluster: average squared distance of its pixels to the image center
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

        # Choose the cluster whose pixels are most centered (smallest mean distance)
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
    def assign_team_color(self, frames, tracks, sample_frames=10):
        # Collect colors from players across several frames
        colors = []
        F = min(sample_frames, len(frames))
        for f in range(F):
            for _, p in tracks["players"][f].items():
                colors.append(self.get_player_color(frames[f], p["bbox"]))
        if len(colors) < 3:
            # Fallback, simple default colors if not enough samples
            self.team_colors = {1:(0,255,255), 2:(255,0,0)}
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
        if np.linalg.norm(cands[0]-cands[1]) < 25:  # distance in BGR space
            # If too close fallback to K=2 clustering
            k2 = KMeans(n_clusters=2, n_init=10, random_state=0).fit(X)
            cands = k2.cluster_centers_

        c1, c2 = cands
        self.team_colors[1] = tuple(map(int, c1))
        self.team_colors[2] = tuple(map(int, c2))

    # Return a temporary team label for the player and collect statistics
    ## How many times a plaxer looks like team 1 or 2
    ## How often a player looks similar to the referee color
    def get_player_team(self, frame, player_bbox, player_id):

        color = self.get_player_color(frame, player_bbox).astype(np.float32)

        # Count how many frames the player was observed
        self.player_obs_counts[player_id] = self.player_obs_counts.get(player_id, 0) + 1

        # Distances to the two team colors
        c1 = np.asarray(self.team_colors[1], dtype=np.float32)
        c2 = np.asarray(self.team_colors[2], dtype=np.float32)
        d1 = np.linalg.norm(color - c1)
        d2 = np.linalg.norm(color - c2)

        # Assign the player to the closer team color
        team = 1 if d1 < d2 else 2

        # Count votes on referee similarity
        if self.referee_color is not None:
            cref = np.asarray(self.referee_color, dtype=np.float32)
            dref = np.linalg.norm(color - cref)

            # Check brightness in HSV to focus in darker referee kits
            col_uint8 = color.astype(np.uint8).reshape(1, 1, 3)
            h, s, v = cv2.cvtColor(col_uint8, cv2.COLOR_BGR2HSV)[0, 0]

            # If referee color is clearly closer than either team color, add a referee vote for the player
            if dref < 0.6 * min(d1, d2):
                prev = self.ref_vote_counts.get(player_id, 0)
                self.ref_vote_counts[player_id] = prev + 1
        
        # Collect team votes for the player
        votes = self.team_vote_counts.get(player_id)
        if votes is None:
            votes = {1: 0, 2: 0}
            self.team_vote_counts[player_id] = votes
        votes[team] += 1

        return team

    # Infer the team 1 or 2 for a bbox using only color
    def infer_team_for_bbox(self, frame, bbox):

        color = self.get_player_color(frame, bbox).astype(np.float32)

        c1 = np.asarray(self.team_colors[1], dtype=np.float32)
        c2 = np.asarray(self.team_colors[2], dtype=np.float32)

        d1 = np.linalg.norm(color - c1)
        d2 = np.linalg.norm(color - c2)

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