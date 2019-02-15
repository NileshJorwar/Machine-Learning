# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 23:35:11 2018

@author: niles
"""

import face_recognition
known_image = face_recognition.load_image_file("img1.jpg")
#face_locations = face_recognition.face_locations(known_image, model="cnn")
unknown_image = face_recognition.load_image_file("img1.jpg")

biden_encoding = face_recognition.face_encodings(known_image)[0]
unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

results = face_recognition.compare_faces([biden_encoding], unknown_encoding)
