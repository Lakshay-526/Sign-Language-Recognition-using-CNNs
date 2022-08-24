import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import os
import pyttsx3
#engine = pyttsx3.init()

# load saved model from PC
model = tf.keras.models.load_model(
    r'/Users/lakshaygupta/Desktop/Sign-Language-Recognition-main/slr_mnist_opt1.h5')
model.summary()
data_dir = '/Users/lakshaygupta/Desktop/Sign-Language-Recognition-main/Gesture Image Pre-Processed Data'
# getting the labels form data directory
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
          'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

# initiating the video source, 0 for internal camera
cap = cv2.VideoCapture(0)

while(True):

    # Capture the video frame
    # by frame
    ret, frame = cap.read()

    # Display the resulting frame

    cv2.rectangle(frame, (100, 100), (500, 500), (0, 0, 255), 5)
    # region of intrest
    roi = frame[100:500, 100:500]
    img = cv2.resize(roi, (28, 28))
    # cv2.imshow('roi', roi)
    img = img/255
    img = np.float32(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    # make predication about the current frame
    prediction = model.predict(img)
    char_index = np.argmax(prediction, axis=1)
    print(char_index, prediction[0, char_index]*100)

    confidence = round(prediction[0, char_index[0]]*100, 1)
    predicted_char = labels[char_index[0]]

    # Initialize the engine
    # engine = pyttsx3.init()
    # engine.say(predicted_char)
    # engine.runAndWait()

    font = cv2.FONT_HERSHEY_TRIPLEX
    fontScale = 1
    color = (0, 255, 255)
    thickness = 2

    # writing the predicted char and its confidence percentage to the frame
    msg = predicted_char + ', Conf: ' + str(confidence)+' %'
    cv2.putText(frame, msg, (80, 80), font, fontScale, color, thickness)

    cv2.imshow('frame', frame)
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
cap.release()
# Destroy all the windows
cv2.destroyAllWindows()
# while(True):

#     _, frame = cap.read()
#     cv2.rectangle(frame, (100, 100), (300, 300), (0, 0, 255), 5)
#     # region of intrest
#     roi = frame[100:300, 100:300]
#     img = cv2.resize(roi, (50, 50))
#     cv2.imshow('roi', roi)

#     img = img/255

#     # make predication about the current frame
#     prediction = model.predict(img.reshape(1, 50, 50, 3))
#     char_index = np.argmax(prediction)
#     # print(char_index,prediction[0,char_index]*100)

#     confidence = round(prediction[0, char_index]*100, 1)
#     predicted_char = labels[char_index]

#     # Initialize the engine
#     engine = pyttsx3.init()
#     engine.say(predicted_char)
#     engine.runAndWait()

#     font = cv2.FONT_HERSHEY_TRIPLEX
#     fontScale = 1
#     color = (0, 255, 255)
#     thickness = 2

#     # writing the predicted char and its confidence percentage to the frame
#     msg = predicted_char + ', Conf: ' + str(confidence)+' %'
#     cv2.putText(frame, msg, (80, 80), font, fontScale, color, thickness)

#     cv2.imshow('frame', frame)

#     # close the camera when press 'q'
#     if cv2.waitKey(10) & 0xFF == ord('q'):
#         break

# release the camera and close all windows
# cap.release()
# cv2.destroyAllWindows()


# -----------------------------------------------------------------------------------------

# import cv2
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# import os
# import pyttsx3
# #engine = pyttsx3.init()


# # load saved model from PC
# model = tf.keras.models.load_model(
#     r'/Users/lakshaygupta/Desktop/Sign-Language-Recognition-main/slr_isl.h5')
# model.summary()
# data_dir = 'Indian'
# # getting the labels form data directory
# labels = sorted(os.listdir(data_dir))
# labels[-1] = 'Nothing'
# print(labels)

# # initiating the video source, 0 for internal camera
# cap = cv2.VideoCapture(0)
# while(True):

#     _, frame = cap.read()
#     cv2.rectangle(frame, (100, 100), (500, 500), (0, 0, 255), 5)
#     # region of intrest
#     roi = frame[100:500, 100:500]
#     img = cv2.resize(roi, (64, 64))
#     cv2.imshow('roi', roi)

#     img = img/255

#     # make predication about the current frame
#     prediction = model.predict(img.reshape(1, 64, 64, 3))
#     char_index = np.argmax(prediction)
#     print(char_index, prediction[0, char_index]*100)

#     confidence = round(prediction[0, char_index]*100, 1)
#     predicted_char = labels[char_index]

#     # Initialize the engine
#     # engine = pyttsx3.init()
#     # engine.say(predicted_char)
#     # engine.runAndWait()

#     font = cv2.FONT_HERSHEY_TRIPLEX
#     fontScale = 1
#     color = (0, 255, 255)
#     thickness = 2

#     # writing the predicted char and its confidence percentage to the frame
#     msg = predicted_char + ', Conf: ' + str(confidence)+' %'
#     cv2.putText(frame, msg, (80, 80), font, fontScale, color, thickness)

#     cv2.imshow('frame', frame)

#     # close the camera when press 'q'
#     if cv2.waitKey(10) & 0xFF == ord('q'):
#         break

# # release the camera and close all windows
# cap.release()
# cv2.destroyAllWindows()
