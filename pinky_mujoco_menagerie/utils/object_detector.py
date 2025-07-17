import numpy as np
from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, model_path="best.pt", conf=0.3):
        self.model = YOLO(model_path)
        self.conf = conf

    def detect(self, image: np.ndarray):
        results = self.model(image, conf=self.conf, device="cpu", verbose=False)
        w = image.shape[1]

        info = {}
        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0].item())
                label = self.model.names[cls]
                x, _, _, _ = map(int, box.xywh[0])
                confidence = round(box.conf[0].item(), 2)
                
                if x < w / 3:
                    direction = "왼쪽"
                elif x > w * 2 / 3:
                    direction = "오른쪽"
                else:
                    direction = "중앙"
                
                # 동일 label에 대해 정보가 없다면 초기화
                if (label not in info) or (confidence > info[label].get("confidence", 0)):
                    info[label] = {"label": label, "direction": direction}
        detection_list = [{"label": v["label"], "direction": v["direction"]} for v in info.values()]
        observation = ", ".join(f"{d['label']}:{d['direction']}" for d in detection_list)
        return observation
    
    def predict(self, img):
        # img: np.ndarray (BGR, OpenCV)
        results = self.model(img, device="cpu", verbose=False)[0]
        result_img = results.plot()
        return result_img