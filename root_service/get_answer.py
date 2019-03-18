import pickle
from keras.preprocessing import image
from keras.models import model_from_json
import numpy as np
import os
import cv2 as cv


class GetAnswerByPath:
    def __init__(self):
        # load model
        json_file = open('../data/trained_model_3_small_new_preproc.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights("../data/trained_model_3_small_new_preproc.h5")
        self.model = loaded_model
        self.input_size = (128, 128, 3)
        self.treshhold = 0.45

        self.frontal_face_cascade = cv.CascadeClassifier('../data/haarcascade_frontalface_default.xml')
        self.profile_face_cascade = cv.CascadeClassifier('../data/haarcascade_profileface.xml')

    # find face using open cv
    def findface(self, img):
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        gray = np.array(gray, dtype='uint8')
        faces = self.frontal_face_cascade.detectMultiScale(gray, 1.3, 5)
        profile_faces = self.profile_face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                img[:y, :] = 0
                img[y + h:, :] = 0
                img[:, :x] = 0
                img[:, x + h:] = 0

        elif len(profile_faces) > 0:
            for (x, y, w, h) in profile_faces:
                cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                img[:y, :] = 0
                img[y + h:, :] = 0
                img[:, :x] = 0
                img[:, x + h:] = 0

        return img

    def get_answer(self, path):
        # score image by path
        for img_path in os.listdir(path):
            img = image.load_img(path + img_path, target_size=(128, 128))
            img = np.array(img, dtype='uint8')
            img = self.findface(img)
            img = image.img_to_array(img) / 255
            score = self.model.predict(img.reshape(1, 128, 128, 3), verbose=0)
            if score > self.treshhold:
                print(path+img_path)

