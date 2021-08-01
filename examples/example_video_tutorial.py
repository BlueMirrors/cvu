"""Example of how to run yolov5 object detection on a video file.
"""
import argparse
from tqdm import tqdm
from vidsz.opencv import Reader, Writer
from cvu.detector import Detector


def process_video(video_source, backend, device, max_frames=float("inf")):
    """Perform Object Detection on Video

    Args:
        video_source (str): video file//webcam to inference
        backend (str): backend to use. Defaults to "onnx".
        device (str): device to use (cpu or gpu or tpu).
        max_frames (int, Optional): maximum number of frames to inference.'.
        Defaults to float('inf').
    """

    # create video reader and writer
    reader = Reader(video_source)
    writer = Writer(reader, name="output.mp4")

    # debug reader-writer info
    print(reader)
    print(writer)

    # create coco detector
    detector = Detector(classes="coco", backend=backend, device=device)

    # process video
    for frame in tqdm(reader):

        # inference
        predictions = detector(frame)

        # draw and write predictions to video
        writer.write(predictions.draw(frame))

        # you can also print predictions, iterate through etc.
        # print(predictions)

        if reader.frame_count > max_frames:
            break

    print("Saved Output File at:", writer.name)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="CVU Video Example1")
    PARSER.add_argument('-video',
                        "-v",
                        type=str,
                        default=None,
                        help="video file//webcam to inference")

    PARSER.add_argument('-backend',
                        "-b",
                        type=str,
                        default="onnx",
                        help="backend to use")

    PARSER.add_argument('-device',
                        "-d",
                        type=str,
                        default="auto",
                        help='cpu or gpu or tpu')

    PARSER.add_argument('-max-frames',
                        "-mf",
                        type=int,
                        default=None,
                        help='maximum number of frames to inference.')

    OPT = PARSER.parse_args()

    if OPT.max_frames is None:
        OPT.max_frames = 500 if OPT.video.isdigit() else float('inf')

    process_video(OPT.video, OPT.backend, OPT.device, OPT.max_frames)
