import json
import math
import sys

import cv2
import numpy
import pandas as pd
from transformers import pipeline
from ultralytics import YOLO


def crop_image(frame, left_bottom_coords: tuple, right_top_coords: tuple):
    # Вырезаем часть изображения по заданным координатам
    cropped_image = frame[left_bottom_coords[1]:right_top_coords[1], left_bottom_coords[0]:right_top_coords[0]]
    return cropped_image


def inPolygon(x, y, xp, yp, w, h):
    c = 0
    for i in range(len(xp)):
        if (((yp[i] * h <= y and y < yp[i - 1] * h) or (yp[i - 1] * h <= y and y < yp[i] * h)) and
                (x > (xp[i - 1] * w - xp[i] * w) * (y - yp[i] * h) / (yp[i - 1] * h - yp[i] * h) + xp[
                    i] * w)): c = 1 - c
    return c


def column(matrix, i):
    return [row[i] for row in matrix]


def proect(p, point1, point2):
    k = (point2[1] - point1[1]) / (point2[1] - point1[1])
    k1 = -1 / k  # y=kx+b b=y-kx
    b = p[1] - k1 * p[0]
    return k1, b


def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])  # Typo was here

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y


def distance(p1, p2):
    return math.sqrt((p2[1] - p1[1]) ** 2 + (p2[0] - p1[0]) ** 2)


argv = sys.argv
# Load the YOLOv8 model
model = YOLO('yolov8l.pt').to(f'cuda:{argv[3]}')
# model = RTDETR('rtdetr-l.pt').to(f'cuda:{argv[3]}')
model.fuse()

classification_pipline = pipeline("image-classification", model="/home/ips/hackathon-2/MobileNetv2/checkpoint-204",
                                  device=7)
# Open the video file
#video_path = rf"./home/nailmarsel/Documents/KRA-44-169-2023-08-17-evening.mp4"
video_paths = [argv[1]]
text_paths = [argv[2]]

file_name = []
car = []
quantity_car = []
average_speed_car = []
van = []
quantity_van = []
average_speed_van = []
bus = []
quantity_bus = []
average_speed_bus = []

for video_path, text_path in zip(video_paths, text_paths):
    cap = cv2.VideoCapture(video_path)
    with open(text_path) as f:
        data = json.load(f)
    areas=data['areas']
    ln = len(areas)
    zones_1=numpy.array(data['zones'][0])
    zones_2=numpy.array(data['zones'][1])
    # print(data['zones'])
    zones_1_1=zones_1[:,0]
    # print(zones_1_1)
    zones_2_1=zones_2[:,0]
    zones_1_2=zones_1[:,1]
    zones_2_2=zones_2[:,1]
    # Loop through the video frames
    counter = 0
    speed_count={} #'id':['start','stop']
    class_count={} #'id':['start','stop']
    flags = {}
    coefs = {}
    max_conf = {}
    frame = {}
    fps = cap.get(cv2.CAP_PROP_FPS)
    while cap.isOpened():
        if counter % 1000 == 0:
            print(counter, '-----------------------')
            # Read a frame from the video
        success, frame = cap.read()
        if success:
            # if counter==1000:
            #     break
            if counter % 4 == 0:
                # Run YOLOv8 tracking on the frame, persisting tracks between frames
                # inp = torch.Tensor(frame).to('cuda').view()
                results = model.track(frame, persist=True, imgsz=(416, 416), classes=[2, 5, 7],
                                      device=f'cuda:{argv[3]}', conf=0.35, verbose=False)
                # annotated_frame = results[0].plot(line_width=2)
                for i in range(len(results[0].boxes.cls)):
                    box = results[0].boxes.xyxy[i]
                    p = (int((box[0] + box[2]) // 2), int((box[1] + box[3]) // 2))
                    # cv2.circle(annotated_frame, (int((box[0]+box[2])//2), int((box[1]+box[3])//2)), 10, (0, 0, 255), -1)
                    h, w = results[0].boxes.orig_shape

                    try:
                        id = int(results[0].boxes.id[i].tolist())
                    except:
                        continue
                    cls = int(results[0].boxes.cls[i].tolist())
                    conf = int(results[0].boxes.conf[i].tolist())
                    # print(box)
                    if id not in max_conf.keys():
                        max_conf[id] = 0

                    if cls == 7 and conf > max_conf[id]:
                        bo = [box[0], box[1] + 1.5 * (box[1] - box[3]), box[2], box[3]]
                        frame1[id] = crop_image(frame, (box[0], box[2]), (box[1], box[3]))
                        max_conf[id] = conf
                    if id not in speed_count.keys():
                        speed_count[id] = [-1, -1]
                        flags[id] = 'False'
                    x = speed_count[id][0]
                    if id not in class_count.keys():
                        class_count[id] = []
                    class_count[id].append(cls)
                    for area in areas:
                        if inPolygon((box[0] + box[2]) / 2, (box[1] + box[3]) / 2, numpy.array(area)[:, 0],
                                     numpy.array(area)[:, 1], w, h) and speed_count[id][0] == -1 and flags[id >

                        point1 = ((area[0][0] + area[3][0]) / 2 * w, (area[0][1] + area[3][1]) / 2 * h)
                        point2 = ((area[1][0] + area[2][0]) / 2 * w, (area[1][1] + area[2][1]) / 2 * h)
                        k, b = proect((int((box[0] + box[2]) // 2), int((box[1] + box[3]) // 2)), point1, point2)
                        x = 0
                        y = b
                        x1, y1 = line_intersection([[x, y], [p[0], p[1]]],
                        [[point1[0], point1[1]], [point2[0], point2[1]]])
                        coef=max([distance([x1, y1], point1), distance([x1, y1], point2)])
                        speed_count[id][0]=counter
                        coefs[id]=(coef/distance(point1, point2))


                        flags[id]=area
                        elif not inPolygon((box[0]+box[2]) / 2, (box[1]+box[3]) / 2, numpy.array(area)[:, 0], numpy.array(area)[:, 1], w, h) and speed_count[id][1] == -1 and fl >

                                                               speed_count[id][1] = counter
                        flags[id] = 'None'
                        break
                        # if id not in class_count.keys():
                        #       class_count[id]=[]
                        # class_count[id].append(cls)
                        # cv2.circle(annotated_frame, (int(area[0][0] * w), int(area[0][1] * h)), 10, (255, 255, 0),
                        #            -1)
                        # cv2.circle(annotated_frame, (int(area[1][0] * w), int(area[1][1] * h)), 10, (255, 255, 0),
                        #            -1)
                        # cv2.circle(annotated_frame, (int(area[2][0] * w), int(area[2][1] * h)), 10, (255, 255, 0),
                        #            -1)
                        # cv2.circle(annotated_frame, (int(area[3][0] * w), int(area[3][1] * h)), 10, (255, 255, 0),
                        #            -1)

                        # Visualize the results on the frame

                        # Display the annotated frame
                        # dim = (int(1920 * 0.5), int(1080 * 0.5))
                        # print(frame_width, frame_height)
                        # print(dim)
                        # resize image

                        # annotated_frame = cv2.resize(annotated_frame, dim, interpolation=cv2.INTER_AREA)
                        # cv2.imshow("YOLOv8 Tracking", annotated_frame)

                        # Break the loop if 'q' is pressed
                        # if cv2.waitKey(1) & 0xFF == ord("q"):
                        # break
                        counter += 1
                    else:
                        # Break the loop if the end of the video is reached
                        break
                    print(class_count, speed_count)
                    print('---------------------')

                    classes = [0, 0, 0]

mean = {'car': [], 'bus': [], 'truck': []}
for key in class_count.keys():
    classe = max(set(class_count[key]), key=class_count[key].count)
    if classe == 2:
        classes[0] += 1
        if speed_count[key][1] != -1 and speed_count[key][0] != -1:
            car_speed = (3.6 * 20 * coefs[key]) / ((speed_count[key][1] - speed_count[key][0]) / fps)
            if car_speed < 80:
                mean['car'].append(car_speed)
    elif classe == 5:

        image = frame[key]
        predicted_label = classification_pipline(image)[0]['label']
        if str(predicted_label) == (0):
            classes[1] += 1

            if speed_count[key][1] != -1 and speed_count[key][0] != -1:
                bus_speed = (3.6 * 20 * coefs[key]) / ((speed_count[key][1] - speed_count[key][0]) / fps)
                if bus_speed < 70:
                    mean['bus'].append(bus_speed)

    else:
        classes[2] += 1
        if speed_count[key][1] != -1 and speed_count[key][0] != -1:
            van_speed = (3.6 * 20 * coefs[key]) / ((speed_count[key][1] - speed_count[key][0]) / fps)
            if van_speed < 70:
                mean['truck'].append(van_speed)

    print(mean)
    print(classes)

    try:
        car_mean = (sum(mean['car']) / len(mean['car']))
    except:
        car_mean = 0.0
    try:
        bus_mean = (sum(mean['bus']) / len(mean['bus']))
    except:
        bus_mean = 0.0
    try:
        van_mean = (sum(mean['truck']) / len(mean['truck']))
    except:
        van_mean = 0.0

    print(car_mean / 2)
    print(bus_mean / 2)
    print(van_mean / 2)

    file_name.append(text_path[text_path.rfind('\\') + 1:text_path.find('.')])
    quantity_car.append(int(classes[0] * 0.85))
    average_speed_car.append(car_mean / 2)
    quantity_van.append(classes[2])
    average_speed_van.append(van_mean / 2)
    quantity_bus.append(classes[1])
    average_speed_bus.append(bus_mean / 2)
    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()

pd.DataFrame({'file_name': file_name, 'quantity_car': quantity_car,
              'average_speed_car': average_speed_car, 'quantity_van': quantity_van,
              'average_speed_van': average_speed_van, 'quantity_bus': quantity_bus,
              'average_speed_bus': average_speed_bus}).set_index(['file_name']).to_csv(f"{file_name[0]}.csv", sep=';')
