"""This module contains core Yolov5 Implementation with various
functional backends.

Find more about Yolov5 here from their official repository
https://github.com/ultralytics/yolov5

Example Usage

    # import core
    from cvu.detector.yolov5 import Yolov5

    # create detector object
    detector = Yolov5(classes='coco')

    # load images in BGR format (np.ndarray) using any tool
    import cv2
    image = cv2.imread('temp/zidane.jpg')

    # inference
    predictions = detector(image)

    # print results
    print(predictions)

    # draw results
    predictions.draw(image)
"""
from .core import Yolov5
