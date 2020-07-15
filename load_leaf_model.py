from keras.preprocessing.image import img_to_array
import imutils
import cv2
import numpy as np
from imutils import paths
import os
from keras.models import load_model

# initialize the list of data and labels
data = []
labels = []
imagePaths = []

for imagePath in sorted(list(paths.list_images("test_set/WithoutMask"))):
    imagePaths.append(imagePath)


imagePaths = np.array(imagePaths)
#idxs = np.random.randint(0, len(imagePaths), size=(10,))
#imagePaths = imagePaths[idxs]

counter=0
for imagePath in imagePaths:
    # load the image, pre-process it, and store it in the data list
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (64,64))
    image = img_to_array(image)
    data.append(image)



# scale the raw pixel intensities to the range [0, 1] .....
data = np.array(data) / 255.0

classLabels = ['With Mask', 'Without Mask']

# load the pre-trained network
print("[INFO] loading pre-trained network...")
model = load_model("mask_googleInception_model.h5")

# make predictions on the images
preds = model.predict(data, batch_size=32).argmax(axis=1)

# loop over the sample images
# load the face detector cascade and smile detector CNN
detector = cv2.CascadeClassifier("cascade.xml")

for (i, imagePath) in enumerate(imagePaths):
    # load the example image, draw the prediction, and display it
    # to our screen
    image = cv2.imread(imagePath)
    imgX = cv2.resize(image, (64, 64))
    imgX = imgX.astype("float") / 255.0
    imgX = img_to_array(imgX)
    imgX = np.expand_dims(imgX, axis=0)
    argMax = np.argmax(model.predict(imgX))
    print("Total-->", model.predict(imgX))

    labelX = classLabels[argMax]

    frame = imutils.resize(image, width=300)
    frameClone = frame.copy()
    # detect faces in the input frame, then clone the frame so that
    # we can draw on it
    rects = detector.detectMultiScale(frame, scaleFactor=1.1,
                                      minNeighbors=5, minSize=(30, 30),
                                      flags=cv2.CASCADE_SCALE_IMAGE)
    print(f"{len(rects)} faces in image")
    for (fX, fY, fW, fH) in rects:
        # extract the ROI of the face from the grayscale image,
        # resize it to a fixed 28x28 pixels, and then prepare the
        # ROI for classification via the CNN
        roi = frame[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (64,64))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        # determine the probabilities of both "smiling" and "not
        # smiling", then set the label accordingly
        ( WithMask,WithoutMask) = model.predict(roi)[0]
        print("face -->",WithoutMask,WithMask)
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
    cv2.putText(frameClone, "Label: {}".format(labelX),
                (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 2)
    cv2.imshow("Face", frameClone)
    cv2.waitKey(0)
