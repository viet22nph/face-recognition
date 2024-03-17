import cv2
import os
import numpy as np

def getImagesAndLabels(path):
    faceSamples = []
    ids = []
    label_map = {}  # Tạo một bản đồ từ tên thư mục sang id
    current_id = 0

    for root, dirs, files in os.walk(path):
        for dir_name in dirs:
            label_map[dir_name] = current_id
            current_id += 1

    for root, dirs, files in os.walk(path):
        for file_name in files:
            if file_name.endswith('.jpg') or file_name.endswith('.png'):
                image_path = os.path.join(root, file_name)
                label = os.path.basename(os.path.dirname(image_path))  # Nhận nhãn từ tên thư mục cha
                if label not in label_map:
                    continue
                id_ = label_map[label]

                image = cv2.imread(image_path)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                faces = faceCascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

                for (x, y, w, h) in faces:
                    roi_gray = gray[y:y + h, x:x + w]
                    faceSamples.append(roi_gray)
                    ids.append(id_)
                cv2.imshow("train data", image)
                cv2.waitKey(100)

    return faceSamples, ids

path = 'datasets'
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

faces, ids = getImagesAndLabels(path)


recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, np.array(ids))
recognizer.save('trainer/trainer.yml')

print("\n[INFO] {0} khuon mat dang duoc traning. thoat".format(len(np.unique(ids))))
