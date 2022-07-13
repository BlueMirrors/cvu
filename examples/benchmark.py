"""Benchmark default CVU models in different backends.
"""
import os
import sys
import importlib
import time
import argparse

import cv2

sys.path.insert(0, "./")
from cvu.detector import Detector
from cvu.utils.backend.package import setup_package
from cvu.utils.google_utils import gdrive_download

BACKEND_FROM_DEVICE = {
    'cpu': ['onnx', 'torch', 'tflite', 'tensorflow'],
    'gpu': ['onnx', 'torch', 'tensorrt', 'tensorflow'],
    'tpu': ['tensorflow']
}

COLOR_MAP = {
    'OK': '\033[92m',
    'ERROR': '\033[91m',
    'RESET': '\033[0m',
    'CYAN': '\033[96m',
    'YELLOW': '\033[93m',
    'HEADER': '\033[95m'
}


def print_benchmark(backend_benchmark: dict) -> None:
    """Print benchmark results.

    Args:
        backend_benchmark (dict): dictionary of backend -> benchmark
    """
    for backend in sorted(backend_benchmark,
                          key=lambda key: -backend_benchmark[key]):
        print(COLOR_MAP['OK'],
              f"FPS({backend}): {round(backend_benchmark[backend], 3)}",
              COLOR_MAP['RESET'])


def install_dependencies() -> None:
    """Install dependencies for benchmarking.
    """
    setup_package("vidsz", "cpu")
    vidsz = importlib.import_module("vidsz.opencv")
    return vidsz


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


def test_image(backends: list, img: str, iterations: int, warmups: int,
               device: str, auto_install: bool) -> None:
    """Benchmark default model of backend with read/write image

    Args:
        backend (list): list of backends
        img (str): path to image
        iterations (int): number of iterations to benchmark for.
        warmups (int): number of iterations for warmup.
        device (str): device to benchmark on.
        auto_install (bool): auto install dependencies
    """
    backend_benchmark = {}
    # download files if needed
    if img == 'zidane.jpg':
        setup_static_files(no_video=True)
        img = 'temp/zidane.jpg'

    # setup
    for backend in backends:
        frame = cv2.imread(img)
        detector = Detector(classes='coco',
                            backend=backend,
                            device=device,
                            auto_install=auto_install)

        # Warm up
        for _ in range(warmups):
            detector(frame)

        # Benchmark inference
        start = time.time()
        for _ in range(iterations):
            detector(frame)
        backend_benchmark[backend] = iterations / (time.time() - start)
        print(COLOR_MAP['OK'],
              f"FPS({backend}): {round(backend_benchmark[backend],3)}",
              COLOR_MAP['RESET'])

        # write output
        detector(frame).draw(frame)
        cv2.imwrite(f'{img.split(".")[0]}_{backend}.jpg', frame)
        print(COLOR_MAP['CYAN'],
              f'Image saved at: {img.split(".")[0]}_{backend}.jpg',
              COLOR_MAP['RESET'])

    # sort benchmarks and print
    print_benchmark(backend_benchmark)


def test_video(backends: list, video: str, max_frames: int, warmups: int,
               device: str, no_write: bool, auto_install: bool) -> None:
    """Benchmark default model of backend with read/write
    video and option to not write output.
    Args:
        backend (list): list of backends
        video (str): path to video
        max_frames (int): number of frames to benchmark for.
        warmups (int): number of iterations for warmup.
        device (str): device to benchmark on.
        no_write (bool): don't write output.
        auto_install (bool): auto install dependencies
    """
    backend_benchmark = {}
    # download files if needed
    if video == 'people.mp4':
        setup_static_files()
        video = 'temp/people.mp4'

    # setup
    vidsz = install_dependencies()

    # loop over backends
    for backend in backends:
        detector = Detector(classes='coco',
                            backend=backend,
                            device=device,
                            auto_install=auto_install)
        reader = vidsz.Reader(video)
        if not no_write:
            writer = vidsz.Writer(reader,
                                  name=f"{video.split('.')[0]}_{backend}.mp4")
        # Warm up
        for frame in reader:
            detector(frame).draw(frame)
            if not no_write:
                writer.write(frame)
            if reader.frame_count > warmups:
                break

        # Benchmark inference
        start = time.time()
        for frame in reader:
            detector(frame).draw(frame)
            if not no_write:
                writer.write(frame)
            if reader.frame_count > max_frames:
                break

        backend_benchmark[backend] = (reader.frame_count -
                                      warmups) / (time.time() - start)
        print(COLOR_MAP['OK'],
              f"FPS({backend}): {round(backend_benchmark[backend],3)}",
              COLOR_MAP['RESET'])
        if not no_write:
            print(COLOR_MAP['CYAN'],
                  f'Video saved at: {video.split(".")[0]}_{backend}.mp4',
                  COLOR_MAP['RESET'])
            writer.release()
        reader.release()

    # sort and print benchmarks
    print_benchmark(backend_benchmark)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="CVU Benchmark")
    PARSER.add_argument('-backend',
                        nargs='+',
                        default=[],
                        help="name(s) of the backend")
    PARSER.add_argument('-img',
                        type=str,
                        default=None,
                        help="image to inference")
    PARSER.add_argument('-video',
                        type=str,
                        default=None,
                        help="video to inference")
    PARSER.add_argument('-device',
                        type=str,
                        help='cpu or gpu or tpu',
                        required=True)
    PARSER.add_argument('-warmups',
                        type=int,
                        default=5,
                        help='number of warmup iters')
    PARSER.add_argument('-iterations',
                        type=int,
                        default=500,
                        help='number of iterations')
    PARSER.add_argument('-max_frames',
                        type=int,
                        default=500,
                        help='number of frames to benchmark for')
    PARSER.add_argument('-no-write',
                        action='store_true',
                        default=False,
                        help='do not write output')
    PARSER.add_argument('-auto-install',
                        action='store_true',
                        help='auto install dependencies')
    OPT = PARSER.parse_args()

    # set default warmup and iterations if device=gpu
    if OPT.device == 'gpu':
        if OPT.warmups < 50:
            OPT.warmups = 50
            print(COLOR_MAP['HEADER'],
                  "For better benchmark results, provide warmups equal to",
                  "or greater than 50. Setting default value as 50.",
                  COLOR_MAP['RESET'])
        if OPT.iterations <= OPT.warmups:
            OPT.iterations = OPT.warmups + 10
            print(
                COLOR_MAP['HEADER'],
                "Iterations should be greater than warmups for better benchmark results.",
                f"Setting default value as {OPT.warmups + 10}",
                COLOR_MAP['RESET'])
    # set default backend if backend not specified
    if not OPT.backend:
        OPT.backend = BACKEND_FROM_DEVICE[OPT.device]
    # run inference based on image or video
    if OPT.img:
        test_image(OPT.backend, OPT.img, OPT.iterations, OPT.warmups,
                   OPT.device, OPT.auto_install)
    elif OPT.video:
        test_video(OPT.backend, OPT.video, OPT.max_frames, OPT.warmups,
                   OPT.device, OPT.no_write, OPT.auto_install)
    else:
        # run inference on default image and videos
        print(COLOR_MAP['OK'],
              "As no -img or -video argument was passed, the tool will",
              "download default image and video and run benchmark on it.",
              COLOR_MAP['RESET'])
        print(COLOR_MAP['YELLOW'], "IMAGE BENCHMARK", COLOR_MAP['RESET'])
        test_image(OPT.backend, "zidane.jpg", OPT.iterations, OPT.warmups,
                   OPT.device, OPT.auto_install)
        print(COLOR_MAP['YELLOW'], "VIDEO BENCHMARK", COLOR_MAP['RESET'])
        test_video(OPT.backend, "people.mp4", OPT.max_frames, OPT.warmups,
                   OPT.device, OPT.no_write, OPT.auto_install)
