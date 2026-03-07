import cv2
import numpy as np
import random


class TrackedPoint:
    def __init__(self, pos, life, size):
        self.pos = np.array(pos, dtype=np.float32)
        self.life = life
        self.size = size


class MotionOverlay:
    def __init__(
        self,
        max_points=80,
        pts_per_frame=6,
        life_frames=12,
        min_size=10,
        max_size=24,
        neighbor_links=2,
        jitter_px=0.3,
        orb_fast_threshold=18,
    ):
        self.max_points = max_points
        self.pts_per_frame = pts_per_frame
        self.life_frames = life_frames
        self.min_size = min_size
        self.max_size = max_size
        self.neighbor_links = neighbor_links
        self.jitter_px = jitter_px

        self.active = []
        self.prev_gray = None
        self.orb = cv2.ORB_create(nfeatures=1200, fastThreshold=orb_fast_threshold)

    def _sample_size(self):
        return random.randint(self.min_size, self.max_size)

    def detect_points(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # 1. Track existing points with optical flow
        if self.prev_gray is not None and self.active:
            prev_pts = np.array([p.pos for p in self.active], dtype=np.float32).reshape(-1, 1, 2)

            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_gray,
                gray,
                prev_pts,
                None,
                winSize=(21, 21),
                maxLevel=3,
            )

            new_active = []
            for tp, new_pt, ok in zip(self.active, next_pts.reshape(-1, 2), status.reshape(-1)):
                if not ok:
                    continue

                x, y = new_pt
                if 0 <= x < w and 0 <= y < h and tp.life > 0:
                    tp.pos = new_pt
                    tp.life -= 1

                    if self.jitter_px > 0:
                        tp.pos += np.random.normal(0, self.jitter_px, size=2)
                        tp.pos[0] = np.clip(tp.pos[0], 0, w - 1)
                        tp.pos[1] = np.clip(tp.pos[1], 0, h - 1)

                    new_active.append(tp)

            self.active = new_active

        # 2. Spawn new points from ORB keypoints
        kps = self.orb.detect(gray, None)
        kps = sorted(kps, key=lambda k: k.response, reverse=True)

        spawned = 0
        for kp in kps:
            if len(self.active) >= self.max_points or spawned >= self.pts_per_frame:
                break

            x, y = kp.pt

            # avoid placing points too close to existing ones
            too_close = any(np.linalg.norm(tp.pos - np.array([x, y])) < 12 for tp in self.active)
            if too_close:
                continue

            self.active.append(
                TrackedPoint(
                    pos=(x, y),
                    life=self.life_frames,
                    size=self._sample_size(),
                )
            )
            spawned += 1

        self.prev_gray = gray

    def draw_overlay(self, frame):
        coords = [tp.pos for tp in self.active]

        # draw connecting lines
        for i, p in enumerate(coords):
            dists = [(j, np.linalg.norm(p - coords[j])) for j in range(len(coords)) if j != i]
            dists.sort(key=lambda x: x[1])

            for j, dist in dists[:self.neighbor_links]:
                if dist < 140:
                    cv2.line(
                        frame,
                        tuple(p.astype(int)),
                        tuple(coords[j].astype(int)),
                        (255, 255, 255),
                        1,
                    )

        # draw boxes
        for tp in self.active:
            x, y = tp.pos
            s = tp.size

            tl = (int(x - s // 2), int(y - s // 2))
            br = (int(x + s // 2), int(y + s // 2))

            # optional inverted box interior for that "pop" effect
            x1, y1 = max(0, tl[0]), max(0, tl[1])
            x2, y2 = min(frame.shape[1], br[0]), min(frame.shape[0], br[1])

            roi = frame[y1:y2, x1:x2]
            if roi.size:
                frame[y1:y2, x1:x2] = 255 - roi

            cv2.rectangle(frame, tl, br, (255, 255, 255), 1)

        return frame