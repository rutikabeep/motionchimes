import cv2
import numpy as np
import random
from collections import deque


class AnalysisHUD:
    def __init__(self, history_len=72):
        self.history = deque(maxlen=history_len)
        self.history_len = history_len
        self.tick = 0

        # BGR palette
        self.line_color = (170, 220, 255)
        self.highlight = (220, 245, 255)
        self.soft_glow = (120, 210, 255)
        self.white = (255, 255, 255)

        # floating particles that move on the HUD arcs
        self.orb_phase = random.uniform(0, 2 * np.pi)

    def update(self, motion_value: float):
        """motion_value should be normalized roughly between 0 and 1."""
        motion_value = float(np.clip(motion_value, 0.0, 1.0))
        self.history.append(motion_value)
        self.tick += 1

    def draw(self, frame):
        h, w = frame.shape[:2]
        overlay = frame.copy()

        self._draw_top_signal_strip(overlay, w, h)
        self._draw_side_measurement_arcs(overlay, w, h)
        self._draw_corner_dots(overlay, w, h)

        # subtle but visible blend
        return cv2.addWeighted(overlay, 0.32, frame, 0.68, 0)

    def _draw_top_signal_strip(self, img, w, h):
        if len(self.history) < 2:
            return

        x0 = int(w * 0.16)
        x1 = int(w * 0.84)
        y0 = int(h * 0.10)
        strip_h = int(h * 0.08)

        # base guide line
        cv2.line(img, (x0, y0 + strip_h), (x1, y0 + strip_h), self.line_color, 1, lineType=cv2.LINE_AA)

        vals = list(self.history)
        xs = np.linspace(x0, x1, len(vals)).astype(int)

        # waveform bars
        for x, v in zip(xs, vals):
            bar_h = int(strip_h * (0.15 + 0.85 * v))
            cv2.line(
                img,
                (x, y0 + strip_h),
                (x, y0 + strip_h - bar_h),
                self.highlight,
                1,
                lineType=cv2.LINE_AA,
            )

        # small circular endpoints
        cv2.circle(img, (x0, y0 + strip_h), 4, self.highlight, 1, lineType=cv2.LINE_AA)
        cv2.circle(img, (x1, y0 + strip_h), 4, self.highlight, 1, lineType=cv2.LINE_AA)

    def _draw_side_measurement_arcs(self, img, w, h):
        center_y = int(h * 0.56)
        radius = int(min(w, h) * 0.18)

        left_cx = int(w * 0.17)
        right_cx = int(w * 0.83)

        # dotted arcs left and right
        self._draw_dotted_arc(img, (left_cx, center_y), radius, -68, 68, dots=12)
        self._draw_dotted_arc(img, (right_cx, center_y), radius, 112, 248, dots=12)

        # tiny floating orb moving along arcs
        self.orb_phase += 0.05
        left_ang = np.deg2rad(-68 + (np.sin(self.orb_phase) * 0.5 + 0.5) * 136)
        right_ang = np.deg2rad(112 + (np.cos(self.orb_phase) * 0.5 + 0.5) * 136)

        lx = int(left_cx + radius * np.cos(left_ang))
        ly = int(center_y + radius * np.sin(left_ang))
        rx = int(right_cx + radius * np.cos(right_ang))
        ry = int(center_y + radius * np.sin(right_ang))

        self._glow_dot(img, lx, ly, 5)
        self._glow_dot(img, rx, ry, 5)

    def _draw_corner_dots(self, img, w, h):
        points = [
            (int(w * 0.18), int(h * 0.24)),
            (int(w * 0.82), int(h * 0.24)),
            (int(w * 0.18), int(h * 0.82)),
            (int(w * 0.82), int(h * 0.82)),
        ]
        for x, y in points:
            cv2.circle(img, (x, y), 2, self.highlight, -1, lineType=cv2.LINE_AA)

    def _draw_dotted_arc(self, img, center, radius, start_deg, end_deg, dots=10):
        cx, cy = center
        angles = np.linspace(np.deg2rad(start_deg), np.deg2rad(end_deg), dots)

        prev = None
        for i, ang in enumerate(angles):
            x = int(cx + radius * np.cos(ang))
            y = int(cy + radius * np.sin(ang))

            cv2.circle(img, (x, y), 2 if i in (0, len(angles) - 1) else 1, self.highlight, -1, lineType=cv2.LINE_AA)

            if prev is not None:
                cv2.line(img, prev, (x, y), self.line_color, 1, lineType=cv2.LINE_AA)

            prev = (x, y)

    def _glow_dot(self, img, x, y, r):
        temp = img.copy()
        cv2.circle(temp, (x, y), r + 4, self.soft_glow, -1, lineType=cv2.LINE_AA)
        img[:] = cv2.addWeighted(temp, 0.12, img, 0.88, 0)

        temp = img.copy()
        cv2.circle(temp, (x, y), r + 1, self.highlight, -1, lineType=cv2.LINE_AA)
        img[:] = cv2.addWeighted(temp, 0.18, img, 0.82, 0)

        cv2.circle(img, (x, y), r // 2 + 1, self.white, -1, lineType=cv2.LINE_AA)