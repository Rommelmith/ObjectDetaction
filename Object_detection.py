import cv2
import numpy as np

class ObjectDetection:
    def __init__(self, weights_path=r'C:\Users\romme\PycharmProjects\ObjectDetection\dnn_model\yolov4.weights', cfg_path=r'C:\Users\romme\PycharmProjects\ObjectDetection\dnn_model\yolov4.cfg'):
        print("Loading Object Detection")
        print("Running Opencv dnn with YOLOV4")
        self.nmsThreshold = 0.4
        self.confThreshold = 0.5
        self.image_size = 608
        net = cv2.dnn.readNet(weights_path, cfg_path)

        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.model = cv2.dnn.DetectionModel(net)

        self.classes = self.load_class_names()
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))

        self.model.setInputParams(size=(self.image_size, self.image_size), scale=1/255)

    def load_class_names(self, classes_path=r"C:\Users\romme\PycharmProjects\ObjectDetection\dnn_model\classes.txt"):
        classes = []
        with open(classes_path, "r") as file_object:
            classes = [line.strip() for line in file_object.readlines()]
        return classes

    def detect(self, frame):
        return self.model.detect(frame, nmsThreshold=self.nmsThreshold, confThreshold=self.confThreshold)
