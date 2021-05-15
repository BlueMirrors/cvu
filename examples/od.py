#utils
import json
from utils import ObjectDetector
from video_utils import LiveStreamReader, VideoWriter

video = LiveStreamReader('https://bluemirrors.github.io/stream/test')
writer = VideoWriter(video, 'output.mp4')

detector = ObjectDetector(classes='person')

for frame in video:
    preds = detector(frame)

    # preds is list of objects.

    # json
    preds_dicty = preds.to_json()  # {...}

    # write as yolo
    print(preds.to_yolo())  # returns string

    # write as coco
    print(preds.to_coco())

    # draw on farme
    print(preds.plot(frame))

    # counts
    print(preds.object_count())  # dict {class:count}

    # filter
    # pred.filter(condition) --> filter_preds

    # index support
    print(preds[0])

    # pop support
    print(preds.pop())

    # print as a list
    print(preds)

    # behave list
    for object_ in preds:
        # json
        object_dict = object_.to_json()  # {...}

        # bbox
        print(object_.bbox)

        # tl, br, cx,cy...
        print((object_.box.tl))

        # object conf (classf * objectness)
        print(object_.confidence)

        # class
        print(object_.class_name)

        # class id
        print(object_.class_id)

        # write as yolo
        print(object_.to_yolo())  # returns string

        # write as coco
        print(object_.to_coco())

        # draw on frame
        print(object_.plot(frame))

        # print as a box (tl-br) and class name
        print(object_)  # id person 0 0 340 430
