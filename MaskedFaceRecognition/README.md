# masked_face_recognition

## models

- `res10_300x300_ssd_iter_140000.caffemodel` - Caffe model weights, used for the face detection
- `deploy.prototxt` - additional file describing the Caffe model
- `shape_predictor_68_face_landmarks.dat` - 68 facial landmark model used for ROI selection
- `cascade_final.h5` - the final model trained on the gathered dataset using Haar Cascade for ROI selection
- `landmark_final.h5` - the final model trained on the gathered dataset using 68 facial landmark for ROI selection

## face_detection.py

A file with a single class called `ROIDetector` used by the recognition and cropping scripts.

## recognize_image.py

It is the script used for a single file recognition. Its output can be seen on the Figure \ref{fig:recognize_image}. The script accepts following arguments:

- `-i/--image*` - a relative path to the image that will be given for the system
- `-b/--base_dataset` - the directory name with the images of the people, used for result image
- `-e/--eye` - a full path to the Haar cascade eye detector
- `-f/--face` - a relative path to the face detection model
- `-r/--recognizer` - a relative path to the face recognition model
- `-c/--confidence` - a confidence for the caffe model to filter out weak detections
- `-l/--landmark` - a relative path to the landmark models
- `-m/--method` - a method for the ROI selection, either `cascade` or `landmark`

## recognize_video.py

Similar script to the previous one except this one uses video as an input. The script accepts following arguments:

- `-b/--base_dataset` - the directory name with the images of the people, used for result image
- `-e/--eye` - a full path to the Haar cascade eye detector
- `-f/--face` - a relative path to the face detection model
- `-r/--recognizer` - a relative path to the face recognition model
- `-c/--confidence` - a confidence for the caffe model to filter out weak detections
- `-l/--landmark` - a relative path to the landmark models
- `-m/--method` - a method for the ROI selection, either `cascade` or `landmark`

## layout.py

A layout file for the PyQt library used in the \emph{recognize_video.py} script.

## train_model.py

A script used for the model training. Necessary comments were left in the code.

## crop_images.py

A script used for the dataset cropping. It iterates over the dataset and crops the ROI with selected algorithm. Necessary comments were left in the code.

## validate_dataset.py

A script that validates the given dataset with selected model. As the output it prints the classification report and a confusion matrix. The script accepts following arguments:

- `-d/--dataset` - a relative path to the dataset to validate
- `-r/--recognizer` - a relative path to the face recognition model
