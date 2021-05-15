from utils import ObjectDetector, Pipeline
from utils.yolov5 import preprocess
from video_utils import LiveStreamReader, VideoWriter

video = LiveStreamReader('https://bluemirrors.github.io/stream/test')
writer = VideoWriter('output.mp4')

# easy-peasy deepstream
pipe = Pipeline(
    source=video,
    process=[preprocess,
             ObjectDetector(classes='person'), postprocess],
    # sink='json'
    # sink=writer,
    sink=None,
    async_run=True)  # return or just kill outputs

pipe.start()

pipe.query()  # --> response if sink is None;

pipe.release()  # kill the process, release the resources
