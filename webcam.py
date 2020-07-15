from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2

# load the face detector cascade and smile detector CNN
detector = cv2.CascadeClassifier("cascade.xml")

model = load_model("mask_googleInception_model.h5")

# if a video path was not supplied, grab the reference to the webcam
camera = cv2.VideoCapture(0)

# keep looping
while True:
    # grab the current frame
    (grabbed, frame) = camera.read()
    # if we are viewing a video and we did not grab a frame, then we
    # have reached the end of the video

    # resize the frame, convert it to grayscale, and then clone the
    # original frame so we can draw on it later in the program
    frame = imutils.resize(frame, width=300)
    gray = frame
    frameClone = frame.copy()
    # detect faces in the input frame, then clone the frame so that
    # we can draw on it
    rects = detector.detectMultiScale(gray, scaleFactor=1.1,
                                      minNeighbors=5, minSize=(30, 30),
                                      flags=cv2.CASCADE_SCALE_IMAGE)
    for (fX, fY, fW, fH) in rects:
        # extract the ROI of the face from the grayscale image,
        # resize it to a fixed 28x28 pixels, and then prepare the
        # ROI for classification via the CNN
        roi = gray[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (64,64))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        # determine the probabilities of both "smiling" and "not
        # smiling", then set the label accordingly
        (WithMask, WithoutMask) = model.predict(roi)[0]
        print(WithMask, WithoutMask)
        label = "Without_MASK" if WithoutMask > WithMask else "With_MASK"

        # display the label and bounding box rectangle on the with_mask
        # frame
        if WithoutMask > WithMask:
            color_ = (0, 0, 255)
        else:
            color_ = (0, 255, 0)

        cv2.putText(frameClone, label, (fX, fY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_, 2)
        cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH), color_, 2)

        # show our detected faces along with smiling/not smiling labels
        cv2.imshow("Face", frameClone)

    # if the ’q’ key is pressed, stop the loop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
