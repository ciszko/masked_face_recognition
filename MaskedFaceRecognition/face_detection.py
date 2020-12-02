import numpy as np
import cv2
import os
import imutils
import dlib
from imutils import face_utils


class ROIDetector:
    def __init__(self, face_model, eye_model, landmark_model, confidence=0.5, method='cascade', draw=False):
        """ROIDetector is a region of interest detector for masked face recognition.
        It uses two models to determine where the uncovered part of the face is.

        Args:\n
            face_model (str): path to the caffe face model\n
            eye_model (str): path to the haar cascade model\n
            landmark_model (str): path to the landmark model\n
            confidence (float): threshold used for face detection\n
            method (str): method for selecting the roi. Can be 'cascade' or 'landmark'\n
            draw (bool): draw the detection boxes on the image
        """
        BASEDIR = os.path.dirname(os.path.realpath(__file__))
        prototxt = os.path.sep.join(
            [BASEDIR, os.path.dirname(face_model), "deploy.prototxt"])
        face_model = os.path.join(BASEDIR, face_model)
        self.net = cv2.dnn.readNet(prototxt, face_model)
        self.eye_cascade = cv2.CascadeClassifier(eye_model)
        self.confidence = confidence
        if landmark_model:
            self.landmark = dlib.shape_predictor(
                os.path.join(BASEDIR, landmark_model))
        else:
            self.landmark = None
        self.method = method
        self.draw = draw

    def get_faces(self, image):
        """returns regions of interest based on the face. If face with two eyes has been found
        then the image is being cut with the bottom line of the detected eyes. If only face has
        been found then function returns the exact half of the image

        Args:\n
            image (np.array): image to look for the faces
        """
        self.image = image.copy()
        self.image_draw = image.copy()
        self.faces = []
        self.roi = []
        h, w, _ = image.shape
        blob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)), 1,
            (300, 300), (132.0, 117.9, 108.9)
        )
        # print("[INFO] Looking for faces on the picture")
        self.net.setInput(blob)
        detections = self.net.forward()
        face_num = 1
        for i in range(0, detections.shape[2]):

            # extract the probability
            confidence = detections[0, 0, i, 2]

            # consider only face with certain threshold
            if confidence > self.confidence:
                # compute the (x, y)-coordinates of the bounding box for the object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                # draw the bounding box of the face along with the associated probability
                if self.draw:
                    text = "{:.2f}%".format(confidence * 100)
                    y = startY - 10 if startY - 10 > 10 else startY + 10
                    cv2.rectangle(self.image_draw, (startX, startY),
                                  (endX, endY), (0, 0, 255), 4)
                    cv2.putText(
                        self.image_draw, text,
                        (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    y = startY - 10 if startY - 10 > 10 else startY + 10
                    cv2.putText(
                        self.image_draw, str(face_num),
                        (endX - 10, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                # extract the face
                self.faces.append(
                    {'id': face_num, 'face': [startY, endY, startX, endX]})
                face_num += 1

    def get_roi_from_face(self, face_roi, face_id):
        """funtion returns the region of interest from the face based on the selected
        method.

        Args:\n
            face_img (np.array): image containing only face (with or without the mask)\n
            face_id (int): id of the face 
        """

        face = self.image_draw[face_roi[0]:face_roi[1], face_roi[2]:face_roi[3], :]
        face_h, face_w, _ = face.shape
        if self.method == 'landmark':
            # create dlib rectangle for the whole face image
            rect = dlib.rectangle(0, 0, face_w, face_h)
            # self.landmark is the facial features detector
            shape = self.landmark(face, rect)
            # convert the shape to numpy array
            shape = face_utils.shape_to_np(shape)
            # loop over the (x, y)-coordinates for the facial landmarks
            # and draw them on the image
            if self.draw:
                for (x, y) in shape[28:46]:
                    cv2.circle(face, (x, y), 2, (0, 255, 0), 5)
                cv2.line(face, (0, (shape[29][1])), (face_w,  (shape[29][1])),
                         (0, 255, 0), thickness=3)
            self.roi.append({'id': face_id, 'roi': self.image[face_roi[0]: (face_roi[0]+shape[29][1]),
                                                              face_roi[2]: face_roi[3], :]})
            # return face[0:shape[29][1], :, :]
        elif self.method == 'cascade':
            # get the upper portion of the face
            top_face = face[0: int(face_h * 0.65), :, :]
            # look for the eyes on the image
            # calculate max and min eye size
            max_eye = (int(face_w / 2.5), int(face_h / 2.5))
            min_eye = (int(face_w / 10), int(face_h / 10))
            eyes = self.eye_cascade.detectMultiScale(
                top_face,
                scaleFactor=1.45,
                minNeighbors=4,
                minSize=min_eye,
                maxSize=max_eye
            )
            # create cut y to determine cut point
            cut_y = 0
            if self.draw:
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(top_face, (ex, ey),
                                  (ex + ew, ey + eh), (0, 255, 255), 2)
            # if more than one is found, filter them
            if len(eyes) > 1:
                best_eyes = self.find_most_similiar_eyes(eyes)
                if self.draw:
                    for (bx, by, bw, bh) in best_eyes:
                        cv2.rectangle(
                            top_face,
                            (bx, by),
                            (bx + bw, by + bh),
                            (255, 100, 0),
                            6,
                        )
                # draw the line on the lower eye y position
                low_eye = max(best_eyes, key=lambda x: x[1])
                cut_y = low_eye[1] + low_eye[3]
            # if there is only one eye
            elif len(eyes) == 1:
                cut_y = eyes[0, 1] + eyes[0, 3]
            else:
                # draw a helping line in the middle of the face
                cut_y = int(face_h // 2)
            if self.draw:
                cv2.line(top_face, (0, cut_y), (face_w,  cut_y),
                         (0, 255, 0), thickness=3)
            self.roi.append({'id': face_id, 'roi': self.image[face_roi[0]:face_roi[0] + cut_y,
                                                              face_roi[2]: face_roi[3], :]})

    def find_most_similiar_eyes(self, eyes):
        """This function finds best matching eyes based on their dimensions and positions

        Args:
            eyes (list of [x,y,w,h]): list of lists containing eye position and dimensions
        """
        eye_sizes = sorted([width * height for _, _, width, height in eyes])
        best_match = []
        min = 99999999
        eyes_list = eyes.tolist()
        # eliminate overlapping eyes
        cut = 0  # cut the eye list, prevent comparing previously compared
        for eye in eyes_list:
            _eyes = eyes.tolist()[cut:]
            for other_eye in eyes_list[cut:]:
                if eye is not other_eye:
                    if self.range_overlap(eye[0], eye[2], other_eye[0], other_eye[2]):
                        _eyes.remove(other_eye)
            # compare eye sizes and choose the best one
            if len(_eyes) == 1 and eye != eyes_list[-1]:
                best_match = _eyes
            eye_sizes = [width * height for _, _, width, height in _eyes]
            for i in range(1, len(eye_sizes)):
                temp = abs(eye_sizes[0] - eye_sizes[i])
                if temp < min:
                    min = temp
                    best_match = [_eyes[0], _eyes[i]]
            cut += 1

        return best_match

    def range_overlap(self, a_min, a_size, b_min, b_size):
        """Neither range is completely greater than other
        """
        return not ((a_min > (b_min+b_size)) or (b_min > (a_min+a_size)))

    def get_roi(self, image):
        self.get_faces(image)
        for face in self.faces:
            self.get_roi_from_face(face['face'], face['id'])
        return self.roi

    def get_image_with_roi(self, image):
        self.get_faces(image)
        for face in self.faces:
            self.get_roi_from_face(face['face'], face['id'])
        return [self.image_draw, self.roi]
