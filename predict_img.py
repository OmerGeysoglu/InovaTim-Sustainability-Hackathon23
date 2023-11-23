'''
import os

from ultralytics import YOLO
import cv2

# Load the trained model
MODEL_PATH = "model/video/best.pt"
model = YOLO(MODEL_PATH)

# Load an image
image_path_to_be_predicted = "data/examples/small_1.jpg"
image = cv2.imread(image_path_to_be_predicted)

# Preprocess the image
preprocessed_image = image.copy()

# Perform object detection
results = model.predict(preprocessed_image)
result = results[0]

for box in result.boxes:
  class_id = result.names[box.cls[0].item()]
  cords = box.xyxy[0].tolist()
  cords = [round(x) for x in cords]
  conf = box.conf[0].item()
  x1, y1, x2, y2 = cords
  if class_id == 'empty':
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.putText(image, f"{class_id} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
  else:
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(image, f"{class_id} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
  
  cv2.imshow('Beer Detection', image)

cv2.waitKey(0)
cv2.destroyAllWindows()'''

import cv2
from ultralytics import YOLO

class BeerDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def load_image(self, image_path):
        return cv2.imread(image_path)

    def predict(self, image):
        results = self.model.predict(image)
        return results[0]

    def draw_boxes(self, image, result):
        for box in result.boxes:
            class_id = result.names[box.cls[0].item()]
            cords = box.xyxy[0].tolist()
            cords = [round(x) for x in cords]
            conf = box.conf[0].item()
            x1, y1, x2, y2 = cords
            color = (0, 255, 0) if class_id != 'empty' else (0, 0, 255)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, f"{class_id} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return image

if __name__ == "__main__":
    detector = BeerDetector("model/video/best.pt")
    image = detector.load_image("data/examples/Screenshot_2.png")
    image = cv2.resize(image,(image.shape[1]//2,image.shape[0]//2))
    result = detector.predict(image)
    image_with_boxes = detector.draw_boxes(image, result)
    cv2.imshow('Beer Detection', image_with_boxes)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
