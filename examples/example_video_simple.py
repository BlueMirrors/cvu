"""Example of how to run yolov5 object detection on a video file with
few lines of code.
"""
import argparse
from tqdm import tqdm
from vidsz.opencv import Reader, Writer
from cvu.detector import Detector


def sample_process_video(video_source):
    """Perform Object Detection on Video
    Args:
        video_source (str): video file//webcam to inference
    """
    detector = Detector(classes="coco")
    with Reader(video_source) as reader, Writer(reader) as writer:
        writer.write_all(
            map(lambda frame: detector(frame).draw(frame), tqdm(reader)))
        print("Output video saved at:", writer.name)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="CVU Video Example Compact")
    PARSER.add_argument('-video',
                        "-v",
                        type=str,
                        default=None,
                        help="video file to inference")
    sample_process_video(PARSER.parse_args().video)
