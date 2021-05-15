import json
from utils import ObjectDetector, benchmark
from video_utils import LiveStreamReader, VideoWriter

video = LiveStreamReader('https://bluemirrors.github.io/stream/test')
writer = VideoWriter('output.mp4')

detector = ObjectDetector(classes='person',
                          tracking=True)  # we will use the default tracker.

# detector = ObjectDetector(classes='person', tracker='centroid')

# detector = ObjectDetector(classes='person',
#                           tracker='deepsort',
#                           tracker_weights='path_to_tracker'
#                           )  # if not given, pick default weights; dlib tracker

for frame in video:
    preds = detector(frame)

    # all objects alive or dead (status).
    json_response = preds.to_json()

    for object_ in preds:
        # print as a box (tl-br) and class name
        print(object_)  # unique_id person 0 0 340 430

        # id is unique, and it'll continue from frame to frame, until reset
        print(object_.id)

    if video.frame_count % 1000:
        # now all tracking info is reset, so id will start from 0
        detector.reset()

    # draw on farme
    writer.write(preds.plot(frame))

# release resources
writer.release()
video.release()
