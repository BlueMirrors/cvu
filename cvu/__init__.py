"""CVU: Computer Vision Utils
Easy to use and consistent interface across various Computer Vision
Use cases such as Object Detection, Segmentation, Tracking etc, on various
backends like TensorRT, TFLite, TensorFlow, PyTorch, ONNX etc. This tool aims at making
computer vision accessible to everyone with or without in-depth Computer
Vision Knowledge. CVU already contains multiple state of the art generalized
models for you to use right out of the box, but it also supports custom weights
and model-architectures.

For example, following code snippet will perform Object Detection for
all the coco classes and create an output video with box drawn.
(CVU uses Yolov5s as default detector)

```python
    from vidsz.opencv import Reader, Writer
    from cvu.detector import Detector

    reader = Reader('temp/people.mp4')
    writer = Writer(reader)
    detector = Detector(classes='coco')

    for frame in reader:
        detector(frame).draw(frame)
        writer.write(frame)
```
"""
__version__ = "0.0.2"
