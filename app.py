import os
import tempfile
from pathlib import Path

import gradio as gr

from pipeline import create_motion_chimes


def process_video(video_file):
    if video_file is None:
        raise gr.Error("Please upload a video first.")

    input_path = Path(video_file)
    suffix = input_path.suffix if input_path.suffix else ".mp4"

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        local_input = tmpdir / f"input{suffix}"
        output_path = tmpdir / "motionchimes_output.mp4"

        # copy uploaded file into temp working area
        with open(input_path, "rb") as src, open(local_input, "wb") as dst:
            dst.write(src.read())

        create_motion_chimes(str(local_input), str(output_path))

        # move output somewhere persistent enough for Gradio to serve
        final_output = Path(tempfile.gettempdir()) / f"motionchimes_result_{os.getpid()}.mp4"
        with open(output_path, "rb") as src, open(final_output, "wb") as dst:
            dst.write(src.read())

    return str(final_output), str(final_output)


with gr.Blocks(title="MotionChimes") as app:
    gr.Markdown(
        """
        # MotionChimes
        Upload a video and transform its motion into shimmering geometry and chime-like sound.
        """
    )

    with gr.Row():
        video_input = gr.Video(label="Upload video")
        video_output = gr.Video(label="Preview output")

    download_file = gr.File(label="Download output")

    run_button = gr.Button("Generate MotionChimes")

    run_button.click(
        fn=process_video,
        inputs=video_input,
        outputs=[video_output, download_file],
    )

app.launch()