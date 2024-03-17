import cv2
import os
import random

def resize_image(image, desired_size):
    old_size = image.shape[:2] # Lấy kích thước cũ của ảnh

    ratio = float(desired_size) / max(old_size) # Tính tỷ lệ resize
    new_size = tuple([int(x * ratio) for x in old_size]) # Kích thước mới

    # Resize ảnh và thêm padding nếu cần
    image = cv2.resize(image, (new_size[1], new_size[0]))
    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    color = [0, 0, 0] # Màu của padding, ở đây là màu đen
    new_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return new_image

# Đường dẫn đến thư mục chứa video
pathVideo = 'video'
lables ={}
# Khởi tạo bộ phát hiện khuôn mặt
faceDetector = cv2.CascadeClassifier("opencv-files/haarcascade_frontalface_alt.xml")

# Lặp qua tất cả các thư mục trong thư mục chứa video
for index, folder in enumerate(os.listdir(pathVideo)):
    # Tạo một bản đồ nhãn từ tên thư mục sang id
    lables[index] = folder
    item_path = os.path.join(pathVideo, folder)
    # Tạo đường dẫn đến thư mục chứa ảnh cho mỗi người
    save_path = os.path.join("datasets", folder)
    # Tạo thư mục nếu nó chưa tồn tại
    os.makedirs(save_path, exist_ok=True)

    # Lặp qua tất cả các video trong thư mục của người đó
    for video in os.listdir(item_path):
        # Khởi tạo bộ đếm và mode
        count = 0
        mode = 0o666
        cap = cv2.VideoCapture(os.path.join(item_path, video))

        # Lặp qua các khung hình trong video
        while cap.isOpened():
            ret, img = cap.read()
            if not ret:
                break
            
            # Lật hình ảnh ngược lại
            img = cv2.flip(img, 1)
            desired_size = 700
            img = resize_image(img, desired_size)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Phát hiện khuôn mặt trong ảnh
            faces = faceDetector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=3, minSize=(30, 30))
            for (x, y, w, h) in faces:
                # Ghi ảnh khuôn mặt vào thư mục tương ứng
                if count <= 110:
                    cv2.imwrite(os.path.join(save_path, f"{index}_{count}.png"), img)
                else:
                    cv2.imwrite(os.path.join("datatest", f"{random.randint(0, 10000)}.png"), img)
                count += 1
            cv2.imshow("camera", img)
            if cv2.waitKey(2) & 0xff == 27 or count == 150:
                break

# Giải phóng bộ nhớ và đóng cửa sổ khi kết thúc
cv2.destroyAllWindows()
