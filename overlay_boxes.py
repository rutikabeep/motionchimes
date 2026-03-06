import cv2
import numpy as np


class MotionOverlay:
    def __init__(self, max_points=50):
        self.points = []
        self.max_points = max_points


    def detect_points(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self.max_points,
            qualityLevel=0.01,
            minDistance=10
        )

        if corners is not None:
            self.points = [tuple(p[0]) for p in corners]


    def draw_overlay(self, frame):

        # draw boxes
        for (x, y) in self.points:
            x = int(x)
            y = int(y)

            size = 15
            cv2.rectangle(
                frame,
                (x-size, y-size),
                (x+size, y+size),
                (255,255,255),
                1
            )

        # draw connecting lines
        for i in range(len(self.points)):
            for j in range(i+1, len(self.points)):

                p1 = self.points[i]
                p2 = self.points[j]

                dist = np.linalg.norm(np.array(p1) - np.array(p2))

                if dist < 120:
                    cv2.line(
                        frame,
                        (int(p1[0]), int(p1[1])),
                        (int(p2[0]), int(p2[1])),
                        (200,200,200),
                        1
                    )

        return frame