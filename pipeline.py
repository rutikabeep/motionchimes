import moviepy.editor as mp
from motion_audio import motion_to_audio
from overlay_boxes import MotionOverlay
import cv2
import os


def apply_overlay(video_path, output_path):

    cap = cv2.VideoCapture(video_path)
    overlay = MotionOverlay()

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        overlay.detect_points(frame)
        frame = overlay.draw_overlay(frame)

        out.write(frame)

    cap.release()
    out.release()


def create_motion_chimes(video_path, output_video="output.mp4"):

    temp_overlay_video = "overlay_temp.mp4"
    audio_path = "generated_chimes.wav"

    print("Applying overlay...")
    apply_overlay(video_path, temp_overlay_video)

    print("Generating motion chimes...")
    motion_to_audio(video_path, audio_path)

    print("Combining audio and video...")

    video = mp.VideoFileClip(temp_overlay_video)
    audio = mp.AudioFileClip(audio_path)

    final = video.set_audio(audio)

    final.write_videofile(output_video, codec="libx264", audio_codec="aac")

    os.remove(temp_overlay_video)
    os.remove(audio_path)

    print("Done!")