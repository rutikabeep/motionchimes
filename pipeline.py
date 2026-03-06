import moviepy.editor as mp
from motion_audio import motion_to_audio
import os


def create_motion_chimes(video_path, output_video="output.mp4"):
    
    audio_path = "generated_chimes.wav"

    print("Generating motion-based chime audio.")
    motion_to_audio(video_path, audio_path)

    print("Combining video with generated audio.")

    video = mp.VideoFileClip(video_path)
    audio = mp.AudioFileClip(audio_path)

    final = video.set_audio(audio)

    final.write_videofile(output_video, codec="libx264", audio_codec="aac")

    os.remove(audio_path)

    print("Done!")