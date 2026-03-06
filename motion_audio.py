import cv2
import numpy as np
from scipy.io.wavfile import write


def extract_motion_signal(video_path):
    cap = cv2.VideoCapture(video_path)

    ret, prev = cap.read()
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    motion = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray, prev_gray)
        motion_energy = np.mean(diff)

        motion.append(motion_energy)

        prev_gray = gray

    cap.release()

    motion = np.array(motion)
    motion = motion / np.max(motion)

    return motion


def generate_chimes(motion, fps=30, sr=44100):
    duration = len(motion) / fps
    audio = np.zeros(int(duration * sr))

    for i, m in enumerate(motion):
        if m > 0.2:
            t = i / fps
            idx = int(t * sr)

            length = int(0.2 * sr)
            freq = np.random.uniform(800, 2000)

            time = np.linspace(0, 0.2, length)
            envelope = np.exp(-10 * time)

            tone = np.sin(2 * np.pi * freq * time) * envelope

            end = min(idx + length, len(audio))
            audio[idx:end] += tone[: end - idx]

    audio /= np.max(np.abs(audio))

    return audio


def motion_to_audio(video_path, output_wav):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    motion = extract_motion_signal(video_path)

    audio = generate_chimes(motion, fps)

    write(output_wav, 44100, audio.astype(np.float32))