import os
import cv2
from ultralytics import YOLO

class VideoObjectDetector:
    def __init__(self, model_path, threshold=0.5):
        self.model = YOLO(model_path)
        self.threshold = threshold

    def process_video(self, input_video_path, output_video_path):
        cap = cv2.VideoCapture(input_video_path)
        ret, frame = cap.read()
        H, W, _ = frame.shape
        out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

        while ret:
            results = self.model(frame)[0]
            self.draw_boxes(frame, results)
            out.write(frame)
            ret, frame = cap.read()

        cap.release()
        out.release()
        cv2.destroyAllWindows()

    def draw_boxes(self, frame, results):
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            if score > self.threshold:
                color = (0, 0, 255) if class_id == 0 else (0, 255, 0)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 4)
                cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 3, cv2.LINE_AA)

if __name__ == "__main__":
    VIDEOS_DIR = os.path.join('.','data', 'examples')
    video_path = os.path.join(VIDEOS_DIR, 'video.mp4')
    video_path_out = '{}_out.mp4'.format(video_path)
    model_path = os.path.join('.', 'model', 'video', 'best.pt')

    detector = VideoObjectDetector(model_path,threshold=0.5)
    detector.process_video(video_path, video_path_out)