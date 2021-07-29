"""This module contains implementation of Yolov5 model
for various backends.

A model (aka backend) basically performs inference on a given input numpy array,
and returns result after performing nms and other backend specific postprocessings.
Expected input may or may not be processed depending on the model's structure and backend's
requirements.
"""
