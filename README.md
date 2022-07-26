# CVU: Computer Vision Utils <img src="https://raw.githubusercontent.com/BlueMirrors/cvu/master/static/logo.png" width="30"> 

[![CodeFactor](https://www.codefactor.io/repository/github/bluemirrors/cvu/badge?s=700eb6a402321377322a7f4c15ebf99055e0c299)](https://www.codefactor.io/repository/github/bluemirrors/cvu) [![stability-alpha](https://img.shields.io/badge/stability-alpha-f4d03f.svg)](https://github.com/mkenney/software-guides/blob/master/STABILITY-BADGES.md#alpha)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/) [![Downloads](https://pepy.tech/badge/cvu-python)](https://pepy.tech/project/cvu-python) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1FvebFw40Bm0bUHWCgS0-iuYp8AKLIfSh?usp=sharing)


<br>

Computer Vision deployment tools for dummies and experts.<br><br>
Whether you are developing an optimized computer vision pipeline or just looking to use some quick computer vision in your project, CVU <img src="https://raw.githubusercontent.com/BlueMirrors/cvu/master/static/logo.png" width="12"> can help! Designed to be used by both the expert and the novice, CVU <img src="https://raw.githubusercontent.com/BlueMirrors/cvu/master/static/logo.png" width="12"> aims at making CV pipelines easier to build and consistent around platforms, devices and models.<br><br>



```bash
pip install cvu-python
```


# Index 📋

- [Getting Started](#cvu--says-hi) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1FvebFw40Bm0bUHWCgS0-iuYp8AKLIfSh?usp=sharing)
- [Why CVU?](https://github.com/BlueMirrors/cvu/wiki)
- [Object Detection (YOLOv5)](https://github.com/BlueMirrors/cvu/wiki/YOLOv5-object-detection) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1FvebFw40Bm0bUHWCgS0-iuYp8AKLIfSh?usp=sharing)
  - [TensorRT](https://github.com/BlueMirrors/cvu/wiki/YOLOv5-TensorRT)
  - [Torch](https://github.com/BlueMirrors/cvu/wiki/YOLOv5-Torch)
  - [ONNX](https://github.com/BlueMirrors/cvu/wiki/YOLOv5-ONNX)
  - [TensorFlow](https://github.com/BlueMirrors/cvu/wiki/YOLOv5-TensorFlow)
  - [TFLite](https://github.com/BlueMirrors/cvu/wiki/YOLOv5-TFLite)
- [Devices (CPU, GPU, TPU)](#devices)
- [Benchmark-Tool (YOLOv5)](https://github.com/BlueMirrors/cvu/wiki/Benchmark-tool)
- [Benchmarks Results (YOLOv5)](https://github.com/BlueMirrors/cvu/wiki/YOLOv5-benchmarking)
- [Precission Accuracy (YOLOv5)](https://github.com/BlueMirrors/cvu/wiki/YOLOv5-object-detection#precission-accuracy-yolov5)
- [Examples](https://github.com/BlueMirrors/cvu/tree/master/examples)
- [References](#references)

<br>

# CVU <img src="https://raw.githubusercontent.com/BlueMirrors/cvu/master/static/logo.png" width="25"> Says Hi!

[Index](#index-)


How many installation-steps and lines of code will you need to run object detection on a video with a TensorRT backend? How complicated is it be to test that pipeline in Colab?<br><br>

With CVU <img src="https://raw.githubusercontent.com/BlueMirrors/cvu/master/static/logo.png" width="12">, you just need the following! No extra installation steps needed to run on Colab, just pip install our tool, and you're all set to go!<br>

```python
from vidsz.opencv import Reader, Writer
from cvu.detector import Detector

# set video reader and writer, you can also use normal OpenCV
reader = Reader("example.mp4")
writer = Writer(reader, name="output.mp4")


# create detector with tensorrt backend having fp16 precision by default
detector = Detector(classes="coco", backend="tensorrt")

# process frames
for frame in reader:

    # make predictions.
    preds = detector(frame)

    # draw it on frame
    preds.draw(frame)

    # write it to output
    writer.write(frame)

writer.release()
reader.release()

```

<br>

Want to use less lines of code? How about this! <br>

```python
from cvu.detector import Detector
from vidsz.opencv import Reader, Writer

detector = Detector(classes="coco", backend="tensorrt")


with Reader("example.mp4") as reader:
    with Writer(reader, name="output.mp4") as writer:
        writer.write_all(map(lambda frame:detector(frame).draw(frame), reader))
```

<br>

Want to switch to non-cuda device? Just set `device="cpu"`, and backend to `"onnx"`, `"tflite"`, `"torch"` or `"tensorflow"`.

<br>

```python
detector = Detector(classes="coco", backend="onnx", device="cpu")
```

<br>

Want to use TPU? Just set `device="tpu"` and choose a supported backend (only `"tensorflow"` supported as of the latest release)

<br>

```python
detector = Detector(classes="coco", backend="tensorflow", device="tpu")
```

You can change devices, platforms and backends as much as you want, without having to change your pipeline.

<br>

# Devices

[Index](#index-)

### Support Info

Following is latest support matrix

| Device | TensorFlow | Torch | TFLite | ONNX | TensorRT |
| ------ | ---------- | ----- | ------ | ---- | -------- |
| GPU    | ✅         | ✅    | ❌     | ✅   | ✅       |
| CPU    | ✅         | ✅    | ✅     | ✅   | ❌       |
| TPU    | ✅         | ❌    | ❌     | ❌   | ❌       |

<br>

### Recommended Backends (in order)

Based on FPS performance and various benchmarks

- GPU: `TensorRT` > `Torch` > `ONNX` > `TensorFlow`
- CPU: `ONNX` > `TFLite` > `TensorFlow` > `Torch`
- TPU: `TensorFlow`

<br><br>

# References

- **_Logo-Attribution_**
  <a href="http://www.freepik.com">Designed by roserodionova / Freepik</a>
- [Yolov5 (Default Object Detection Model)](https://github.com/ultralytics/yolov5)
