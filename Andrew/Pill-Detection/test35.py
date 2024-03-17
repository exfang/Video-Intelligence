# Do not use this python script on mainstream detection. 
# This script does not convert the image into square, resulting in wrong location of bounding boxes.

import cv2
import numpy as np
import os
import yaml
from yaml.loader import SafeLoader

class YOLO_Pred():
    def __init__(self, onnx_model, datayaml):
        
        with open('data.yaml', mode='r') as f: # Load the data.yaml file which contains the different pill classes.
            data_yaml = yaml.load(f, Loader=SafeLoader)
        
        self.labels = data_yaml['names'] # Retrieve the classes.
        self.nc = data_yaml['nc'] # Retrieve the number of classes.

#         self.yolo = cv2.dnn.readNetFromONNX('./Model/weights/best.onnx') # Load YOLO Model trained on Google Collab.
        self.yolo = cv2.dnn.readNetFromONNX('./Model/LD_Model/best.onnx')
        self.yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.yolo.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    
    def predictions(self, image, name='Tablet'): # Detect and Predict the image classes. Unknown label with 'Tablet'
        
        row, col, d = image.shape

        # Step 1: Convert image into square image (array)
#         max_rc = max(row, col)
#         input_image = np.zeros((max_rc, max_rc,3), dtype=np.uint8)
#         input_image[0:row, 0:col] = image

        input_image = np.zeros((row, col,3), dtype=np.uint8)
        input_image[0:row, 0:col] = image

        # Step 2: Get prediction from square array
        INPUT_WH_YOLO = 640
        blob = cv2.dnn.blobFromImage(input_image, 1/255, (INPUT_WH_YOLO, INPUT_WH_YOLO), swapRB=True, crop=False)
        self.yolo.setInput(blob)
        preds = self.yolo.forward()

        # Non Maximum Suppression
        # Step 1: Filter detection based on confidence and probability score.
        
        detections = preds[0]
        boxes = []
        confidences = []
        classes = []

        # Width and height of the image
        image_w, image_h = input_image.shape[:2]
        
        
        x_factor = image_w / INPUT_WH_YOLO
        y_factor = image_h / INPUT_WH_YOLO
        
        # Retrieve bounding box values from the detections if confidence and probability score >0.5
        for i in range(len(detections)):
            row = detections[i]
            confidence = row[4]  # confidence of detecting an object
            if confidence > 0.5:
                class_score = row[5:].max()  # maximum probability
                class_id = row[5:].argmax()

                if class_score > 0.5:
                    cx, cy, w, h = row[0:4]

                    # construct bounding box from four values
                    left = int((cx - 0.5 * w) * x_factor)
                    top = int((cy - 0.5 * h) * y_factor)
                    width = int(w * x_factor)
                    height = int(h * y_factor)

                    box = np.array([left, top, width, height])
                    
                    # append values into the list
                    confidences.append(confidence)
                    boxes.append(box)
                    classes.append(class_id)

        # clean
        boxes_np = np.array(boxes).tolist()
        confidences_np = np.array(confidences).tolist()

        # NMS
        index = np.array(cv2.dnn.NMSBoxes(boxes_np, confidences_np, 0.5, 0.5)).flatten()
        normalized_bbox = []
        filtered_class = []
        
        # Draw Bounding Box
        for ind in index:
            x, y, w, h = boxes_np[ind] # Filtered bounding boxes
            filtered_class.append(classes[ind])
            bb_conf = (confidences_np[ind] * 100)
#             classes_id = classes[ind]
#             class_name = self.labels[classes_id]
#             colors = self.generate_colors(classes_id)
            
            # normalize the bounding box coordinates
            cx_normalized = ((x+x+w)/2)/image_w
            cy_normalized = ((y+y+h)/2)/image_h
            width_normalized = w/image_w
            height_normalized = h/image_h
            
            normalized_bbox.append((cx_normalized, cy_normalized, width_normalized, height_normalized))
            # cv2.rectangle(image, start_point, end_point, color, thickness)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
        return image, normalized_bbox, filtered_class

    def generate_colors(self, ID):
        np.random.seed(10)
        colors = np.random.randint(100, 255, size=(self.nc, 3)).tolist()
        return tuple(colors[ID])
