from glob import glob
import numpy as np
import cv2
import os
from scipy.ndimage import rotate
import argparse
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from face_detection import ROIDetector
import tensorflow as tf
from tensorflow.python.keras.applications.vgg16 import preprocess_input


BASEDIR = os.path.dirname(os.path.realpath(__file__))


def plot_image(predictions_array, img):
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img)

    predicted_label = np.argmax(predictions_array)
    color = 'red'

    plt.xlabel("{} {:2.0f}%".format(class_names[predicted_label],
                                    100*np.max(predictions_array)),
               color=color)


def plot_value_array(predictions_array):
    plt.grid(False)
    plt.title("Last CNN's layer output")
    plt.xticks(range(0, len(predictions_array), 5))
    plt.yticks([])
    thisplot = plt.bar(range(len(predictions_array)),
                       predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')


################ argument parser ####################
ap = argparse.ArgumentParser()
ap.add_argument(
    "-i",
    "--image",
    type=str,
    help="path to input image",
)
ap.add_argument(
    "-b",
    "--base_dataset",
    type=str,
    default="train_images",
    help="name of the base dataset containing images of people for the result image",
)
ap.add_argument(
    "-e",
    "--eye",
    type=str,
    default=cv2.data.haarcascades + "haarcascade_eye.xml",
    help="path to cascade eye detector",
)
ap.add_argument(
    "-f",
    "--face",
    type=str,
    default="./models/res10_300x300_ssd_iter_140000.caffemodel",
    help="path to face detector model",
)
ap.add_argument(
    "-r",
    "--recognizer",
    type=str,
    default="models/cascade_final.h5",
    help="path to trained face recognizer model",
)
ap.add_argument(
    "-c",
    "--confidence",
    type=float,
    default=0.5,
    help="minimum probability to filter weak face detections",
)
ap.add_argument(
    '-l',
    '--landmark',
    type=str,
    default="./models/shape_predictor_68_face_landmarks.dat",
    help='path to landmark face predictor'
)
ap.add_argument(
    '-m',
    '--roi_method',
    type=str,
    default="cascade",
    help='roi detection method'
)
args = vars(ap.parse_args())


############### initialize models #####################
roi_detector = ROIDetector(
    args['face'], args['eye'], args['landmark'], args['confidence'], args['roi_method'], draw=True)
print("[INFO] Loading face recognizer...")
try:
    model = load_model(os.path.join(BASEDIR, args["recognizer"]))
except Exception as e:
    print(f"[ERROR] Could not load face recognizer: {e}")
    raise
print("[INFO] Loaded face recognizer succesfully")
######################################################
# Reads all the folders in which images are present

dir_names = glob(os.path.join(
    BASEDIR, args['base_dataset'] + "/*/*"), recursive=True)
dir_names.sort(key=lambda x: x.split('\\')[-2])
class_names = [name.split('\\')[-2] for name in dir_names]
name_id_map = dict(zip(range(len(class_names)), class_names))
res_images = []
for img_path in dir_names:
    pth = "\\".join(img_path.split('\\')[:-1])
    first_img = os.listdir(pth)[0]
    res_images.append(os.path.join(pth, first_img))

#################################
# load the image
image = cv2.imread(args['image'])
# get the ROI
img, roi = roi_detector.get_image_with_roi(image)
face = roi[0]['roi']
# preprocess the image
face = tf.convert_to_tensor(face, dtype=tf.float32)
face = tf.image.resize(face, (224, 224))
face = tf.expand_dims(face, axis=0) / 255.0

predictions = model.predict(face)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.figure()
plt.subplots_adjust(hspace=0.3, wspace=0.1, left=0.05,
                    right=0.95, top=0.95, bottom=0.05)
plt.subplot(2, 2, 1)
plot_image(predictions[0], img)
plt.subplot(2, 2, 2)
best = np.argmax(predictions[0])
img_name = res_images[best]
result = plt.imread(img_name)
plt.grid(False)
plt.xticks([])
plt.yticks([])
plt.imshow(result)
plt.subplot(2, 1, 2)
plot_value_array(predictions[0])
plt.show()
