import torch
from ultralytics import YOLO
import cv2
import time
import numpy as np
import yaml
import os

def getNames():
    default_path = "****************"
    file_data = f"{default_path}/data.yaml"

    with open(file_data, 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)

    names_list = data['names'] if 'names' in data else []

    return names_list

import uuid

def crop_object_region(image, box, output_path):
    x, y, w, h = box
    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x + w, image.shape[1]), min(y + h, image.shape[0])
    cropped_img = image[y1:y2, x1:x2]
    cv2.imwrite(output_path, cropped_img)
    return cropped_img

names_list = getNames()

model = YOLO(model="best.pt")

torch_model = torch.load('best.pt')

size = int(torch_model['train_args']['imgsz'])

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model.to(device)

cap = cv2.VideoCapture(0)

total_frames = 0
start_time = time.time()

# Cria a pasta para salvar as imagens recortadas se ainda não existir
if not os.path.exists('detected_objects'):
    os.makedirs('detected_objects')

while True:
    ret, frame = cap.read()

    detections = model.predict(frame, max_det=1, imgsz=size, conf=0.25)

    for detection in detections:
        boxes = detection.boxes.cpu().numpy()
        if len(boxes.cls) != 0:
            detected_classes = boxes.cls
            count_objects = 0

            for obj in detected_classes:
                class_predict = names_list[int(obj)]
                box = boxes.xywh[count_objects]
                x, y, w, h = box

                x = int(x)
                y = int(y)
                w = int(w)
                h = int(h)

                x = int((x - w/2))
                y = int((y - h/2))

                color = (255, 0, 0)
                thickness = 2

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)

                text = class_predict
                font = cv2.FONT_HERSHEY_SIMPLEX
                bottomLeftCornerOfText = (x, y - 10)
                fontScale = 0.75
                fontColor = (255, 40, 40)
                thickness = 1
                lineType = 2

                cv2.putText(frame, text, bottomLeftCornerOfText, font,
                            fontScale, fontColor, thickness, lineType)
                
                # Gera um UUID v4 (aleatório)
                generated_uuid = uuid.uuid4()
                # Converte o UUID para uma string
                uuid_str = str(generated_uuid)

                output_path = f"detected_objects/{class_predict}_{uuid_str}.jpg"
                crop_object_region(frame, (x, y, w, h), output_path)

                count_objects += 1

    cv2.imshow("Object", frame)
    total_frames += 1

    elapsed_time = time.time() - start_time

    if elapsed_time >= 1:
        fps = total_frames / elapsed_time
        print("FPS:", round(fps, 2))

        total_frames = 0
        start_time = time.time()

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
