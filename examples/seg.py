#utils
from utils import ObjectDetector
from video_utils import LiveStreamReader, VideoWriter

video = LiveStreamReader('https://bluemirrors.github.io/stream/test')
writer = VideoWriter(video, 'output.mp4')

detector = ObjectDetector(classes='person', tracker=True, segment=True)

for frame in video:
    preds = detector(frame)

    # draw on farme
    writer.write(preds.plot(frame))

    # behave list
    for object_ in preds:
        # json
        object_dict = object_.to_json()  # {...}

        # return seg mask for the box
        print(object_.mask)

# release resources
writer.release()
video.release()

# cv_utils
# video_utils
# data_utils
# cv_training_utils
# data_analytics_utils
