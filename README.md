# <img src="static/logo.png" width="30"> CVU: Computer Vision Utils

![status](https://img.shields.io/pypi/status/ansicolortags.svg) [![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)

<br>
CV tools for dummies.

```bash
pip install cvu-python
```

Object Detector

- YoloV5

<br>
Backends (in-development)

- TensorRT
- TFLite
- ONNX
- TensorFlow
- PYTorch

# Detect Objects

```python
from vidsz.opencv import Reader, Writer
from cvu.detector import Detector

# set video reader and writer, you can also use normal OpenCV
reader = Reader("static/example.mp4")
writer = Writer(reader)


# create detector
# by default it'll load pretrained YoloV5-small coco model,
# and filter based on given classes
detector = Detector('coco', backend='onnx')

# print video info: width, height, fps etc.
print(reader)

# print detector info: input_shape, backend, expected_fps, thresholds etc.
print(detector)

# read frame with for loop
for frame in reader:

    # make predictions.
    preds = detector(frame)

    # draw it on frame
    preds.draw(frame)

    # write it to output
    writer.write(frame)

# release
reader.release()
writer.release()

```

**_Logo-Attribution_**
<a href="http://www.freepik.com">Designed by roserodionova / Freepik</a>
