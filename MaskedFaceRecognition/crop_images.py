import os
from face_detection import ROIDetector
from glob import glob
import cv2
import random

BASEDIR = os.path.dirname(os.path.realpath(__file__))

# the dataset to be cropped
DATASET = 'datasets\\AFDB_masked_face_dataset'


def is_ascii(s):
    return all(ord(c) < 128 for c in s)


# initialize the ROIDetector
roi_detector = ROIDetector(
    "face_model/res10_300x300_ssd_iter_140000.caffemodel",
    cv2.data.haarcascades + "haarcascade_eye.xml",
    "./face_model/shape_predictor_68_face_landmarks.dat",
    0.5,
    'cascade',
    True
)

dir_names = glob(os.path.join(
    BASEDIR, DATASET + '/*/*'), recursive=True)

for img_path in dir_names:
    if not is_ascii(img_path):
        extension = img_path.split('.')[-1]
        splitted = img_path.split('\\')
        rand_string = (random.sample(range(0, 99), 10))
        new_file = ''.join([str(x) for x in rand_string]) + '.' + extension
        new_path = '\\'.join(splitted[:-1]) + '\\' + new_file
        os.rename(img_path, new_path)
        img_path = new_path
    image = cv2.imread(img_path)
    roi = roi_detector.get_roi(image)[0]['roi']
    cv2.imwrite(img_path, roi)
