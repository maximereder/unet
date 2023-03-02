import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import pandas as pd
import cv2
from glob import glob
from tqdm import tqdm
import argparse
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
from metrics import dice_loss, dice_coef, iou

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--source", dest="source", help="Image source path.", default='')
parser.add_argument("-o", "--output", dest="output", help="Output source path.", default='')
parser.add_argument("-m", "--model", dest="model", help="Model path.", default='model/model.h5')
parser.add_argument("-is", "--imgsz", dest="imgsz", help="Image size for inference.", default=[304, 3072], nargs='+')
args = parser.parse_args()

H = int(args.imgsz[0])
W = int(args.imgsz[1])

def create_dir(path):
    """Creates a directory if it does not exist.

    Args:   
        path (str): Path to the directory to create.
    """
    if not os.path.exists(path):
        os.makedirs(path)

def read_image(path):
    """Reads and processes an image.

    Args:
        path (str): Path to the image file.

    Returns:
        tuple: Original image and processed image.
    """
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (W, H))
    ori_x = x
    x = x/255.0
    x = x.astype(np.float32)
    return ori_x, x

if __name__ == "__main__":
    print("Start inference")

    """ Load the model """
    with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef, 'dice_loss': dice_loss}):
        model = tf.keras.models.load_model(args.model)

    """ Predict the mask """
    image = cv2.imread(args.source)
    image = cv2.resize(image, (W, H))
    image = image/255.0
    mask = np.squeeze(model.predict(np.expand_dims(image, axis=0), verbose=0))
    mask = np.where(mask > 0.8, 255, 0)
    mask = mask.astype(np.uint8)

    """ Save the predicted mask """
    cv2.imwrite(args.output, mask)

    print("Predicted mask for image saved to: " + args.output)
