import os
from grace import Config
from keras.models import load_model, model_from_json


def get_model():
    """
    """
    return CenterModel()


class CenterModel:
    """
    """
    def __init__(self, session=None):
        self.num_channels = 3
        self.image_height = 112
        self.image_width = 96
        self.model = load_model(os.path.join(Config.ROOT, 'face_model_caffe_converted.h5'), compile=False)

    def predict(self, im):
        return self.model(im)