import torch
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QFileDialog
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
import yaml


def scale_coords(img_size, coords, img_shape):
    # img_size: Tuple (width, height) of original image size
    # coords: Tensor of shape (N, 4) containing bounding box coordinates in the original image space
    # img_shape: Tuple (height, width) of resized image shape

    height_ratio = img_shape[0] / img_size[1]
    width_ratio = img_shape[1] / img_size[0]
    coords[:, 0] *= width_ratio  # x1
    coords[:, 1] *= height_ratio  # y1
    coords[:, 2] *= width_ratio  # x2
    coords[:, 3] *= height_ratio  # y2

    return coords


class YOLOv5Detector:
    def __init__(self, weights, conf_thresh=0.5, nms_thresh=0.5, yaml_path='custom.yaml'):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(weights, map_location=self.device)
        self.model = checkpoint["model"]
        self.model.to(self.device).eval()
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.yaml_path = yaml_path
        self.class_names = self.load_class_names()

    def load_class_names(self):
        with open(self.yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        return data['names']

    def detect(self, image_path):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        model = torch.hub.load("ultralytics/yolov5",
                               "custom", path="best.pt", force_reload=True)
        model.classes = self.class_names

        result = model(img)

        bbox_raw = result.xyxy[0][0]

        bbox = []
        for bound in bbox_raw:
            bbox.append(int(bound.item()))
        bbox = bbox[:4]

        new_image = img.copy()
        cv2.rectangle(new_image, (bbox[0], bbox[1]),
                      (bbox[2], bbox[3]), (255, 0, 0), 5)

        # label = f"{result.names[0]} {bbox_raw[4]:.2f}"
        # cv2.putText(new_image, label, (bbox[0], bbox[1] - 10),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        img = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
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
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "", "Image Files (*.jpg *.jpeg *.png *.bmp)", options=options)
        if file_name:
            self.image_path = file_name
            pixmap = QPixmap(file_name)
            self.label_image.setPixmap(pixmap.scaled(
                self.label_image.width(), self.label_image.height(), Qt.KeepAspectRatio))

    def detect_objects(self):
        if not self.image_path:
            return
        detector = YOLOv5Detector(
            'best.pt')
        image = detector.detect(self.image_path)
        cv2.imwrite('output.jpg', image)
        pixmap = QPixmap('output.jpg')
        self.label_image.setPixmap(pixmap.scaled(
            self.label_image.width(), self.label_image.height(), Qt.KeepAspectRatio))


if __name__ == "__main__":
    app = QApplication([])
    main_window = MainWindow()
    main_window.show()
    app.exec_()
