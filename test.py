import cv2
import os

# Đọc mô hình nhận dạng khuôn mặt
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')

# Load bộ phát hiện khuôn mặt
faceDetector = cv2.CascadeClassifier("opencv-files/haarcascade_frontalface_alt.xml")

# Đọc nhãn từ thư mục datasets
labels = {}
for index, folder in enumerate(os.listdir("datasets")):
    labels[index] = folder

# Hàm nhận dạng khuôn mặt từ một hình ảnh
def recognize_face(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceDetector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=3, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        # Nhận dạng khuôn mặt và tính toán mức độ tự tin
        id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
        label = labels[id]
        if confidence < 70:
            confidence = "{0}%".format(round(100 - confidence))
        else:
            label = "unknown"
            confidence = "{0}%".format(round(100 - confidence))

        # Vẽ hộp và nhãn khuôn mặt nhận dạng được lên hình ảnh
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, label, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(image, str(confidence), (x + 5, y + h - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    return image

# Đường dẫn đến thư mục chứa ảnh kiểm tra
pathTest = "datatest"

# Lặp qua tất cả các ảnh trong thư mục kiểm tra và hiển thị kết quả nhận dạng
for file_name in os.listdir(pathTest):
    image_path = os.path.join(pathTest, file_name)
    result_image = recognize_face(image_path)
    cv2.imshow("Test data", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


