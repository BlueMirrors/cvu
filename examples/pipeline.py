#utils
from utils import ObjectDetector
from video_utils import LiveStreamReader, VideoWriter

video = LiveStreamReader('https://bluemirrors.github.io/stream/test')
writer = VideoWriter(video, 'output.mp4')

detector = ObjectDetector(classes='person')  # roi;

for frame in video:
    preds = detector(frame)

    # draw on farme
    writer.write(preds.plot(frame))

# release resources
writer.release()
video.release()