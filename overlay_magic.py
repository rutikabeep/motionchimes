import cv2
import numpy as np
import random
import math
import colorsys


class MagicAnchor:
    def __init__(self, pos, life, strength, style):
        self.pos = np.array(pos, dtype=np.float32)
        self.life = life
        self.max_life = life
        self.strength = strength
        self.style = style
        self.drift = np.random.normal(0, 0.18, size=2).astype(np.float32)
        self.phase = random.uniform(0, math.tau)
        self.size = int(np.clip(14 + strength * 22 + random.randint(-2, 6), 12, 42))

        # geometry metadata
        self.has_geometry = random.random() < 0.38
        self.geo_shape = random.choice(["octagon", "hexagon", "circle"])
        self.geo_rotation = random.uniform(0, math.tau)
        self.geo_speed = random.uniform(-0.035, 0.035)
        self.geo_scale = random.choice([1.0, 1.618, 2.0])
        self.tick_count = random.randint(3, 6)


class MagicOverlay:
    def __init__(
        self,
        max_anchors=36,
        pts_per_frame=4,
        life_frames=22,
        neighbor_links=2,
        jitter_px=0.14,
        orb_fast_threshold=16,
        motion_threshold=16,
        motion_min_area=55,
    ):
        self.max_anchors = max_anchors
        self.pts_per_frame = pts_per_frame
        self.life_frames = life_frames
        self.neighbor_links = neighbor_links
        self.jitter_px = jitter_px
        self.motion_threshold = motion_threshold
        self.motion_min_area = motion_min_area

        self.prev_gray = None
        self.active = []
        self.last_motion_value = 0.0
        self.orb = cv2.ORB_create(nfeatures=1500, fastThreshold=orb_fast_threshold)

        # one cohesive pastel palette per run
        self.palette = self._generate_pastel_palette()

        self.white = (255, 255, 255)
        self.soft_gold = random.choice(self.palette)
        self.deep_gold = random.choice(self.palette)
        self.line_color = random.choice(self.palette)
        self.geo_color = random.choice(self.palette)
        self.geo_highlight = random.choice(self.palette)
        self.fib_sizes = [8, 13, 21, 34, 55]

    def detect_points(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        if self.prev_gray is None:
            self.last_motion_value = 0.0

        motion_mask = None
        motion_regions = []

        if self.prev_gray is not None:
            diff = cv2.absdiff(self.prev_gray, gray)
            raw_motion = float(np.mean(diff))
            self.last_motion_value = min(1.0, raw_motion / 40.0)

            blur = cv2.GaussianBlur(diff, (7, 7), 0)
            _, thresh = cv2.threshold(blur, self.motion_threshold, 255, cv2.THRESH_BINARY)
            thresh = cv2.dilate(thresh, np.ones((3, 3), np.uint8), iterations=2)
            motion_mask = thresh

            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < self.motion_min_area:
                    continue
                x, y, bw, bh = cv2.boundingRect(contour)
                motion_regions.append((x, y, bw, bh, area))

        # track existing anchors
        if self.prev_gray is not None and self.active:
            prev_pts = np.array([a.pos for a in self.active], dtype=np.float32).reshape(-1, 1, 2)
            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_gray,
                gray,
                prev_pts,
                None,
                winSize=(21, 21),
                maxLevel=3,
            )

            new_active = []
            for anchor, new_pt, ok in zip(self.active, next_pts.reshape(-1, 2), status.reshape(-1)):
                if not ok:
                    continue

                x, y = new_pt
                if not (0 <= x < w and 0 <= y < h):
                    continue

                anchor.pos = new_pt
                anchor.life -= 1
                anchor.phase += 0.14
                anchor.geo_rotation += anchor.geo_speed

                if self.jitter_px > 0:
                    anchor.pos += np.random.normal(0, self.jitter_px, size=2)
                    anchor.pos[0] = np.clip(anchor.pos[0], 0, w - 1)
                    anchor.pos[1] = np.clip(anchor.pos[1], 0, h - 1)

                if anchor.life > 0:
                    new_active.append(anchor)

            self.active = new_active

        # spawn new anchors only in moving areas
        if motion_mask is not None and motion_regions:
            kps = self.orb.detect(gray, None)
            filtered = []
            for kp in kps:
                x, y = kp.pt
                xi, yi = int(x), int(y)
                if 0 <= xi < w and 0 <= yi < h and motion_mask[yi, xi] > 0:
                    filtered.append(kp)

            filtered = sorted(filtered, key=lambda k: k.response, reverse=True)

            spawned = 0
            for kp in filtered:
                if len(self.active) >= self.max_anchors or spawned >= self.pts_per_frame:
                    break

                x, y = kp.pt
                too_close = any(np.linalg.norm(a.pos - np.array([x, y])) < 18 for a in self.active)
                if too_close:
                    continue

                strength = float(np.clip(kp.response * 6.0, 0.25, 1.0))
                style = random.choices(
                    ["chrome_star", "chrome_star", "glint_cluster", "orbit", "blob"],
                    weights=[0.42, 0.18, 0.20, 0.10, 0.10],
                    k=1,
                )[0]

                life = self.life_frames + random.randint(-3, 8)
                self.active.append(MagicAnchor((x, y), life, strength, style))
                spawned += 1

        self.prev_gray = gray

    def draw_overlay(self, frame):
        base = frame.copy()
        line_layer = frame.copy()
        glow = frame.copy()

        coords = [a.pos for a in self.active]

        # faint motion structure lines
        for i, p in enumerate(coords):
            dists = [(j, np.linalg.norm(p - coords[j])) for j in range(len(coords)) if j != i]
            dists.sort(key=lambda x: x[1])

            for j, dist in dists[:self.neighbor_links]:
                if dist < 165:
                    p1 = tuple(p.astype(int))
                    p2 = tuple(coords[j].astype(int))
                    cv2.line(line_layer, p1, p2, self.line_color, 1, lineType=cv2.LINE_AA)

        # occasional triangle/mesh accent
        for i in range(0, len(coords) - 2, 3):
            p1 = coords[i]
            p2 = coords[i + 1]
            p3 = coords[i + 2]

            if (
                np.linalg.norm(p1 - p2) < 150
                and np.linalg.norm(p2 - p3) < 150
                and np.linalg.norm(p1 - p3) < 170
            ):
                pts = np.array(
                    [
                        [int(p1[0]), int(p1[1])],
                        [int(p2[0]), int(p2[1])],
                        [int(p3[0]), int(p3[1])],
                    ],
                    dtype=np.int32,
                )
                cv2.polylines(line_layer, [pts], True, random.choice(self.palette), 1, lineType=cv2.LINE_AA)

        base = cv2.addWeighted(line_layer, 0.18, base, 0.82, 0)

        # render anchors
        for a in self.active:
            x, y = int(a.pos[0]), int(a.pos[1])
            t = max(0.18, a.life / max(1, a.max_life))
            pulse = 0.95 + 0.22 * math.sin(a.phase)
            size = max(10, int(a.size * pulse))

            if a.has_geometry:
                self._draw_geometry(glow, x, y, size, a, t)

            if a.style == "chrome_star":
                self._draw_chrome_star(glow, x, y, size, t)
            elif a.style == "orbit":
                self._draw_orbit(glow, x, y, size, t)
            elif a.style == "glint_cluster":
                self._draw_glint_cluster(glow, x, y, size, t)
            elif a.style == "blob":
                self._draw_blob(glow, x, y, size, t)

        out = cv2.addWeighted(glow, 0.82, base, 0.18, 0)
        return out

    def _generate_pastel_palette(self, n=8):
        palette = []

        family = random.choice([
            0.08,  # warm peach/yellow
            0.16,  # yellow
            0.32,  # mint
            0.55,  # sky blue
            0.68,  # lavender
            0.92,  # pink
        ])

        for _ in range(n):
            h = (family + random.uniform(-0.08, 0.08)) % 1.0
            s = random.uniform(0.22, 0.48)
            v = random.uniform(0.93, 1.0)

            r, g, b = colorsys.hsv_to_rgb(h, s, v)
            palette.append((int(b * 255), int(g * 255), int(r * 255)))

        # make sure each palette keeps some contrast
        palette.append((255, 255, 255))
        palette.append((235, 245, 255))
        return palette

    def _glow_circle(self, img, x, y, radius, color, weight):
        temp = img.copy()
        cv2.circle(temp, (x, y), max(2, radius), color, -1, lineType=cv2.LINE_AA)
        img[:] = cv2.addWeighted(temp, weight, img, 1 - weight, 0)

    def _draw_chrome_star(self, img, x, y, size, alpha):
        glow_a = random.choice(self.palette)
        glow_b = random.choice(self.palette)

        self._glow_circle(img, x, y, int(size * 1.45), glow_a, 0.10 * alpha)
        self._glow_circle(img, x, y, int(size * 1.00), glow_b, 0.14 * alpha)
        self._glow_circle(img, x, y, int(size * 0.55), (255, 255, 255), 0.18 * alpha)

        arm = max(7, int(size * 0.78))
        diag = max(5, int(arm * 0.65))

        cv2.line(img, (x - arm, y), (x + arm, y), self.deep_gold, 4, lineType=cv2.LINE_AA)
        cv2.line(img, (x, y - arm), (x, y + arm), self.deep_gold, 4, lineType=cv2.LINE_AA)

        cv2.line(img, (x - arm, y), (x + arm, y), self.soft_gold, 2, lineType=cv2.LINE_AA)
        cv2.line(img, (x, y - arm), (x, y + arm), self.soft_gold, 2, lineType=cv2.LINE_AA)

        cv2.line(img, (x - diag, y - diag), (x + diag, y + diag), self.white, 1, lineType=cv2.LINE_AA)
        cv2.line(img, (x - diag, y + diag), (x + diag, y - diag), self.white, 1, lineType=cv2.LINE_AA)

        cv2.circle(img, (x, y), 2, self.white, -1, lineType=cv2.LINE_AA)

    def _draw_orbit(self, img, x, y, size, alpha):
        temp = img.copy()

        major = self._quantized_radius(size, 1.1)
        minor = max(4, int(major * 0.42))
        axes = (major, minor)

        angle = int((math.degrees(math.sin(alpha * 4 + x * 0.01)) + 30) % 180)

        orbit_color = random.choice(self.palette)
        cv2.ellipse(temp, (x, y), axes, angle, 0, 360, orbit_color, 2, lineType=cv2.LINE_AA)
        cv2.ellipse(temp, (x, y), axes, angle, 0, 360, self.white, 1, lineType=cv2.LINE_AA)

        self._draw_tiny_glint(temp, x, y, max(5, size // 3), self.white)
        img[:] = cv2.addWeighted(temp, 0.55 * alpha, img, 1 - 0.55 * alpha, 0)

    def _draw_glint_cluster(self, img, x, y, size, alpha):
        self._draw_tiny_glint(img, x, y, max(7, size // 2), random.choice(self.palette))

        for _ in range(random.randint(3, 5)):
            ang = random.uniform(0, math.tau)
            rad = random.uniform(size * 0.45, size * 1.05)
            sx = int(x + math.cos(ang) * rad)
            sy = int(y + math.sin(ang) * rad)
            s = random.randint(4, max(6, size // 2))
            col = random.choice(self.palette)
            self._draw_tiny_glint(img, sx, sy, s, col)

    def _draw_blob(self, img, x, y, size, alpha):
        color = random.choice(self.palette)
        temp = img.copy()
        axes = (max(7, int(size * 0.60)), max(6, int(size * 0.46)))
        angle = random.randint(0, 179)

        cv2.ellipse(temp, (x, y), axes, angle, 0, 360, color, -1, lineType=cv2.LINE_AA)
        cv2.ellipse(temp, (x, y), axes, angle, 0, 360, self.white, 1, lineType=cv2.LINE_AA)

        hx, hy = int(x - axes[0] * 0.25), int(y - axes[1] * 0.2)
        cv2.circle(temp, (hx, hy), max(2, size // 6), self.white, -1, lineType=cv2.LINE_AA)

        img[:] = cv2.addWeighted(temp, 0.50 * alpha, img, 1 - 0.50 * alpha, 0)

    def _draw_tiny_glint(self, img, x, y, size, color):
        arm = max(4, size)
        diag = max(2, int(arm * 0.62))
        cv2.line(img, (x - arm, y), (x + arm, y), color, 1, lineType=cv2.LINE_AA)
        cv2.line(img, (x, y - arm), (x, y + arm), color, 1, lineType=cv2.LINE_AA)
        cv2.line(img, (x - diag, y - diag), (x + diag, y + diag), self.white, 1, lineType=cv2.LINE_AA)
        cv2.line(img, (x - diag, y + diag), (x + diag, y - diag), self.white, 1, lineType=cv2.LINE_AA)
        cv2.circle(img, (x, y), 1, self.white, -1, lineType=cv2.LINE_AA)

    def _draw_geometry(self, img, x, y, size, anchor, alpha):
        radius = self._quantized_radius(size, anchor.geo_scale)

        geo_color = random.choice(self.palette)
        geo_highlight = random.choice(self.palette)

        self._glow_circle(img, x, y, int(radius * 1.05), geo_color, 0.04 * alpha)

        if anchor.geo_shape == "octagon":
            pts = self._regular_polygon_points(x, y, radius, 8, anchor.geo_rotation)
            cv2.polylines(img, [pts], True, geo_color, 1, lineType=cv2.LINE_AA)

        elif anchor.geo_shape == "hexagon":
            pts = self._regular_polygon_points(x, y, radius, 6, anchor.geo_rotation)
            cv2.polylines(img, [pts], True, geo_color, 1, lineType=cv2.LINE_AA)

        else:
            cv2.circle(img, (x, y), radius, geo_color, 1, lineType=cv2.LINE_AA)

        arc_r = max(radius + 4, radius)
        start_a = int((math.degrees(anchor.geo_rotation) + 20) % 360)
        cv2.ellipse(img, (x, y), (arc_r, arc_r), 0, start_a, start_a + 42, geo_highlight, 1, lineType=cv2.LINE_AA)
        cv2.ellipse(img, (x, y), (arc_r, arc_r), 0, start_a + 140, start_a + 182, geo_highlight, 1, lineType=cv2.LINE_AA)

        for i in range(anchor.tick_count):
            ang = anchor.geo_rotation + i * (math.tau / anchor.tick_count)
            r1 = radius + 2
            r2 = radius + 7
            x1 = int(x + math.cos(ang) * r1)
            y1 = int(y + math.sin(ang) * r1)
            x2 = int(x + math.cos(ang) * r2)
            y2 = int(y + math.sin(ang) * r2)
            cv2.line(img, (x1, y1), (x2, y2), geo_highlight, 1, lineType=cv2.LINE_AA)

    def _nearest_fib_size(self, value):
        return min(self.fib_sizes, key=lambda x: abs(x - value))

    def _quantized_radius(self, size, scale=1.0):
        raw = size * scale
        return self._nearest_fib_size(raw)

    def _regular_polygon_points(self, cx, cy, radius, sides, rotation):
        pts = []
        for i in range(sides):
            ang = rotation + (math.tau * i / sides)
            px = int(cx + math.cos(ang) * radius)
            py = int(cy + math.sin(ang) * radius)
            pts.append([px, py])
        return np.array(pts, dtype=np.int32)