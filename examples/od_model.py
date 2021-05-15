import json
from utils import ObjectDetector, benchmark
from video_utils import LiveStreamReader, VideoWriter

video = LiveStreamReader('https://bluemirrors.github.io/stream/test')
writer = VideoWriter('output.mp4')

# detector = ObjectDetector(classes='person', pick_best=True)
# detector = ObjectDetector(classes='person', backend='onnx')
detector = ObjectDetector(classes='person',
                          model='all',
                          backend='all',
                          benchmark=True,
                          dataset='PATH_TO_DATASET',
                          save_config=True)

benchmark(
    detector,
    input_source=video)  # speed, --> best config for best inference time.

benchmark(detector, ground_truth='dataset')  # check for speed and accuracy
