
import numpy as np
import os
import argparse
from tensorflow.keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.applications.vgg16 import preprocess_input
import tensorflow as tf
from sklearn import metrics


BASEDIR = os.path.dirname(os.path.realpath(__file__))

################ argument parser ####################
ap = argparse.ArgumentParser()
ap.add_argument(
    "-d",
    "--dataset",
    type=str,
    default='datasets/test_landmark',
    help="path to dataset directory, relative to this file",
)
ap.add_argument(
    "-r",
    "--recognizer",
    type=str,
    default="models/landmark_final.h5",
    help="path to trained face recognizer model",
)
args = vars(ap.parse_args())


############### initialize models #####################
try:
    model = load_model(os.path.join(BASEDIR, args["recognizer"]))
except Exception as e:
    print(f"[ERROR] Could not load face recognizer: {e}")
    raise
print("[INFO] Loaded face recognizer succesfully")
######################################################

test_datagen = ImageDataGenerator(
    rescale=1/255.,
    preprocessing_function=preprocess_input)

test_generator = test_datagen.flow_from_directory(
    directory=os.path.join(BASEDIR, args['dataset']),
    class_mode=None,
    shuffle=False,
    target_size=(224, 224)
)

preds = model.predict_generator(test_generator)
preds_cls_idx = preds.argmax(axis=-1)

classes = list(range(0, len(preds_cls_idx)))

idx_to_cls = {v: k for k, v in test_generator.class_indices.items()}
preds_cls = np.vectorize(idx_to_cls.get)(preds_cls_idx)
filenames_to_cls = list(zip(test_generator.filenames, preds_cls))

report = metrics.classification_report(
    classes, preds_cls_idx, target_names=preds_cls)
print('[INFO] Classification report')
print(report)

print('[INFO] Confusion matrix')
tf.print(tf.math.confusion_matrix(classes, preds_cls_idx), summarize=-1)
