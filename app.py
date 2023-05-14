import torch
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QFileDialog
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

class YOLOv5Detector:
    def __init__(self, weights, conf_thresh=0.5, nms_thresh=0.5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model  = torch.load('yolov5s.pt', map_location=torch.device('cpu'))
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh

    def detect(self, image_path):
        img = cv2.imread(image_path)
        results = self.model(img)
        results.render()
        for det in results.xyxy[0]:
            if det[4] > self.conf_thresh:
                x1, y1, x2, y2 = det[:4].tolist()
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                label = f"{results.names[int(det[5])]} {det[4]:.2f}"
                cv2.putText(img, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        return img

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLOv5 Object Detection")
        self.setGeometry(100, 100, 800, 600)
        self.image_path = None

        self.label_image = QLabel(self)
        self.label_image.setGeometry(10, 10, 780, 480)
        self.label_image.setAlignment(Qt.AlignCenter)
        self.label_image.setStyleSheet("border: 1px solid black;")

        self.button_browse = QPushButton("Browse Image", self)
        self.button_browse.setGeometry(10, 510, 120, 30)
        self.button_browse.clicked.connect(self.browse_image)

        self.button_detect = QPushButton("Detect Objects", self)
        self.button_detect.setGeometry(140, 510, 120, 30)
        self.button_detect.clicked.connect(self.detect_objects)

    def browse_image(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Image Files (*.jpg *.jpeg *.png *.bmp)", options=options)
        if file_name:
            self.image_path = file_name
            pixmap = QPixmap(file_name)
            self.label_image.setPixmap(pixmap.scaled(self.label_image.width(), self.label_image.height(), Qt.KeepAspectRatio))

    def detect_objects(self):
        if not self.image_path:
            return
        detector = YOLOv5Detector('/Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/site-packages/PyQt5/best.pt')
        image = detector.detect(self.image_path)
        cv2.imwrite('output.jpg', image)
        pixmap = QPixmap('output.jpg')
        self.label_image.setPixmap(pixmap.scaled(self.label_image.width(), self.label_image.height(), Qt.KeepAspectRatio))

if __name__ == "__main__":
    app = QApplication([])
    main_window = MainWindow()
    main_window.show()
    app.exec_()
