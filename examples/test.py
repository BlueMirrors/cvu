"""Sample CVU usage.
"""
import time
import os
import argparse

import cv2
from vidsz.opencv import Reader, Writer

from cvu.detector import Detector

from cvu.utils.google_utils import gdrive_download


def setup_static_files(no_video: bool = False) -> None:
    """Setup needed directories, video and image files for test.

    Args:
        no_video (bool, optional): don't download video. Defaults to False.
    """
    if not os.path.exists("temp"):
        os.mkdir('temp')
    if not os.path.exists("temp/zidane.jpg"):
        gdrive_download("181Htf9x0HVxyZoXYgjwqdZuItCcw_C2n", "temp/zidane.jpg")

    if not no_video and not os.path.exists("temp/people.mp4"):
        gdrive_download("1rioaBCzP9S31DYVh-tHplQ3cgvgoBpNJ", "temp/people.mp4")


def no_write_inference(backend: str,
                       iterations: int = 500,
                       warmups: int = 10) -> None:
    """Benchmark default model of backend without read/write

    Args:
        backend (str): name of backend
        iterations (int, optional): number of iterations to benchmark for. Defaults to 500.
        warmups (int, optional): number of iterations for warmup. Defaults to 10.
    """
    # download files if needed
    setup_static_files(no_video=True)

    # setup
    frame = cv2.imread('temp/zidane.jpg')
    detector = Detector(classes='coco', backend=backend)

    # Warm up
    for _ in range(warmups):
        detector(frame)

    # benchmark
    start = time.time()
    for _ in range(iterations):
        detector(frame)
    delta = time.time() - start
    print(f"FPS-NW({backend}): ", (iterations) / delta)

    # write output
    detector(frame).draw(frame)
    cv2.imwrite(f'temp/zidane_out_{backend}.jpg', frame)


def video_inference(backend: str,
                    quick: bool = False,
                    max_iterations: int = 500,
                    warmups: int = 5) -> None:
    """Benchmark default model of backend with read/write video

    Args:
        backend (str): name of backend
        quick (bool, optional): skip video and only inference single image. Defaults to False.
        max_iterations (int, optional): number of max-iterations to benchmark for. Defaults to 500.
        warmups (int, optional): number of iterations for warmup. Defaults to 5.
    """
    # download files if needed
    setup_static_files()

    # setup
    detector = Detector(classes='coco', backend=backend)

    # inference on video
    if not quick:
        reader = Reader('temp/people.mp4')
        writer = Writer(reader, name=f"temp/people_out_{backend}.mp4")

        # Warm up
        for frame in reader:
            detector(frame).draw(frame)
            writer.write(frame)
            if reader.frame_count > warmups:
                break

        start = time.time()
        for frame in reader:
            detector(frame).draw(frame)
            writer.write(frame)
            if reader.frame_count > max_iterations:
                break

        delta = time.time() - start
        print(f"FPS({backend}): ", (reader.frame_count - warmups) / delta)
        writer.release()
        reader.release()

    # inference on image
    frame = cv2.imread('temp/zidane.jpg')
    preds = detector(frame)
    preds.draw(frame)
    cv2.imwrite(f'temp/zidane_out_{backend}.jpg', frame)
    print(preds)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('-backend',
                        type=str,
                        default='onnx',
                        help="Name of the backend.")
    PARSER.add_argument('-no-write', action='store_true')
    PARSER.add_argument('-quick', action='store_true')

    PARSER.add_argument('-warmups',
                        type=int,
                        default=5,
                        help='number of warmup iters')

    PARSER.add_argument('-iterations',
                        type=int,
                        default=500,
                        help='number of iterations')

    OPT = PARSER.parse_args()
    if OPT.no_write:
        no_write_inference(OPT.backend, OPT.iterations, OPT.warmups)
    else:
        video_inference(OPT.backend, OPT.quick, OPT.iterations, OPT.warmups)
