import cv2

def bgr_to_rgb(image):
    return image[..., ::-1]


def hwc_to_whc(image):
    return image.transpose(2, 0, 1)


def normalize(image):
    return image / 255.0


def resize(image, shape):
    return cv2.resize(image, shape)