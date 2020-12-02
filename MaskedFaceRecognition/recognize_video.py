import qdarkstyle
import cv2
import sys
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import QMainWindow
from tensorflow.keras.models import load_model
import tensorflow as tf
from layout import *
from glob import glob
import os
from face_detection import ROIDetector
import argparse
import numpy as np


BASEDIR = os.path.dirname(os.path.realpath(__file__))
################ argument parser ####################
ap = argparse.ArgumentParser()
ap.add_argument(
    "-b",
    "--base_dataset",
    type=str,
    default="train_images",
    help="name of the base dataset containing images of people",
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
    default="models/res10_300x300_ssd_iter_140000.caffemodel",
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
run_args = vars(ap.parse_args())


class MyMaskedClassifier(QThread):
    changePixmap = pyqtSignal(QImage)
    recognized_person = pyqtSignal(int)
    confidence_signal = pyqtSignal(int)
    graph_signal = pyqtSignal(np.ndarray, int)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.roi_detector = ROIDetector(
            run_args['face'], run_args['eye'], run_args['landmark'], run_args['confidence'], run_args['roi_method'], draw=True)
        self.model = load_model(os.path.join(BASEDIR, run_args["recognizer"]))
        # names related to id
        self.names = sorted(os.listdir(os.path.join(BASEDIR, 'images')))

    def run(self):

        # iniciate id counter
        self.id = 0

        # Initialize and start realtime video capture
        self.cam = cv2.VideoCapture(0)
        self.cam.set(3, 400)  # set video widht
        self.cam.set(4, 400)  # set video height

        while True:

            self.ret, frame = self.cam.read()
            frame = cv2.flip(frame, 1)

            self.frame, self.roi = self.roi_detector.get_image_with_roi(frame)
            if len(self.roi) > 0:
                face = self.roi[0]['roi']
                if face.size > 0:
                    face = tf.convert_to_tensor(face, dtype=tf.float32)
                    face = tf.image.resize(face, (224, 224))
                    face = tf.expand_dims(face, axis=0) / 255.0

                    predictions = self.model.predict(face)[0]
                    max_value = (np.argmax(predictions))
                    confidence = 100*(predictions[max_value])
                    self.graph_signal.emit(predictions, max_value)
                    self.confidence_signal.emit(int(confidence))
                    self.recognized_person.emit(max_value)

            if self.ret:
                rgbImage = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgbImage.shape
                bytesPerLine = ch * w
                convertToQtFormat = QImage(
                    rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888
                )
                p = convertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
                self.changePixmap.emit(p)


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setupUi(self)
        ############## functions ######################
        # load first image from the directory
        self.result_image_src = glob(os.path.join(
            BASEDIR, run_args['base_dataset'] + '/*/*'))[0]

        cv_img = cv2.imread(self.result_image_src)
        # convert the image to Qt format
        qt_img = self.convert_cv_qt(cv_img)
        # display it
        self.result_image.setPixmap(qt_img)
        self.camera_thread = MyMaskedClassifier()
        self.camera_thread.changePixmap.connect(self.setImage)
        self.camera_thread.recognized_person.connect(self.recognized_person)
        self.camera_thread.confidence_signal.connect(self.show_percentage)
        self.camera_thread.graph_signal.connect(self.graph_data)
        self.camera_thread.start()
        self.people = []
        for person_dir in sorted(os.listdir(os.path.join(BASEDIR, run_args['base_dataset']))):
            img_name = os.listdir(os.path.join(
                BASEDIR, run_args['base_dataset'], person_dir))[0]
            img = cv2.imread(os.path.join(BASEDIR, run_args['base_dataset'],
                                          person_dir, img_name))
            img = cv2.resize(img, (301, 371))
            self.people.append(img)

    @pyqtSlot(QImage)
    def setImage(self, image):
        self.camera_video.setPixmap(QPixmap.fromImage(image))

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(
            rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888
        )
        p = convert_to_Qt_format.scaled(
            self.result_frame.width(), self.result_frame.height(), Qt.KeepAspectRatio
        )
        return QPixmap.fromImage(p)

    def recognized_person(self, person):
        # convert the image to Qt format
        img = self.people[person]
        qt_img = self.convert_cv_qt(img)
        self.result_image.setPixmap(qt_img)

    def show_percentage(self, percentage):
        self.confidence_bar.setValue(percentage)

    def graph_data(self, data, max_value):
        self.graph.update_canvas(data, max_value)


if __name__ == "__main__":

    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet())
    window = MainWindow()
    window.show()
    app.exec()
