import json
from utils import ObjectDetector, benchmark
from video_utils import LiveStreamReader, VideoWriter

video = LiveStreamReader('https://bluemirrors.github.io/stream/test')
writer = VideoWriter('output.mp4')

detector = ObjectDetector(classes='person', benchmark=True)
# detector = ObjectDetector(classes='person', backend='onnx')
# detector = ObjectDetector(classes='person',
#                           backend=['tensorrt', 'onnx', 'tensorflow'])

# we will use priority list

# if gpu is there
# if nvidia-gpu is there

# if amd-gpu is there --> TVM support (WebApps)

# ARM --> TFLite else ONNX

# 1. tensorrt  --> try importing;  if works, then this is our backend, else continue with the list
#

benchmark(
    detector,
    input_source=video)  # speed, --> best config for best inference time.

benchmark(detector, ground_truth='dataset')  # check for speed and accuracy
