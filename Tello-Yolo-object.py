import cv2
import numpy as np
from djitellopy import tello
import time
import math
import datetime

#ff = open("fly-log.txt", "a")
#ff.writelines("========= Start on " + str(datetime.datetime.now()) + "===========\n")

whT = 320
confThreshold = 0.5
nmsThreshold = 0.3

classesFile = 'YOLOv3/coco.names'
classNames = []
with open(classesFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
# print(classNames)

modelConfiguration = 'YOLOv3/yolov3.cfg'
modelWeights = 'YOLOv3/yolov3.weights'

net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
# --Tello flying------
me = tello.Tello()
me.connect()
print(me.get_battery())
me.streamon()
me.takeoff()
me.send_rc_control(0, 0, 25, 0)
time.sleep(2)
# configuration
w, h = 360, 240
fbRange = [15000, 17000]
pid = [0.4, 0.4, 0]
pError = 0
# calculation for angular velocity
radius = 100  # 1 meter away
t_per_round = 16  # time per round in second
ang_vel = 2*math.pi/(t_per_round)*radius  # angular velocity
x_vel = int( ang_vel*math.cos(45))
x_vel = -x_vel
y_vel = int(ang_vel*math.sin(45))


def fly_circle(me):
    me.send_rc_control(0, 100, 0, -50)
    me.send_rc_control(0, 100, 0, -50)
    for i in range(10):
        time.sleep(1)
        me.send_rc_control(0, 45, 0, 45)
        print("i={}".format(i))


def find_objects(outputs, img):
    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []

    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w, h = int(det[2] * wT), int(det[3] * hT)
                x, y = int((det[0] * wT) - w / 2), int((det[1] * hT) - h / 2)
                bbox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence))
                # print(len(bbox))
    indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)
    myObjectC = [0, 0]
    myObjectArea = 0
    for i in indices:
        i = i[0]
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
        cv2.putText(img, f'{classNames[classIds[i]].upper()} {int(confs[i] * 100)}%',
                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        if classNames[classIds[i]].upper() == 'PERSON':
            cx = x + w // 2
            cy = y + h // 2
            area = w * h
            #print("cx", cx, "cy", cy, "area", area)
            myObjectC = [cx, cy]
            myObjectArea = area

    return img, [myObjectC, myObjectArea]


def track_person( info, w, pid, pError):
    area = info[1]
    x, y = info[0]
    fb = 0
    lr = 0
    error = x - w // 2
    speed = pid[0] * error + pid[1] * (error - pError)
    speed = int(np.clip(speed, -100, 100))
    curve = 0
    if area > fbRange[0] and area < fbRange[1]:
        # make it fly circle
        #fb = 30 #y_vel
        #speed = 30 #x_vel
        #error = pError  # make it continue
        curve = 1
        fly_circle(me)
    elif area > fbRange[1]:
        fb = -20
    elif area < fbRange[0] and area != 0:
        fb = 20
    if x == 0:
        speed = 0
        error = 0

    print(lr, fb, speed, error, pError)
    #ff.writelines("{}: {}, {}, {}, {}, {}, {}\n".format(datetime.datetime.now(), lr,fb, speed, error, pError, area))
    me.send_rc_control(lr, fb, 0, speed)
    #qqtime.sleep(0.5)
    #if curve == 1:
        #me.rotate_counter_clockwise(15)
    return error

#cap = cv2.VideoCapture(0)
while True:
    #_, img = cap.read()
    img = me.get_frame_read().frame
    img = cv2.resize(img, (w, h))
    #--yolo--
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
    net.setInput(blob)
    layerNames = net.getLayerNames()
    outputNames = [layerNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    # print(outputNames)
    output = net.forward(outputNames)
    img, info = find_objects(output, img)
    pError = track_person( info, w, pid, pError)
    # print("Center", info[0], "Area", info[1])
    cv2.imshow("Output", img)
    #if me.get_height() > 300:
        #me.send_rc_control(0, 0, -20, 10)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        me.land()
        print(me.get_battery())
        #ff.writelines("Battery {}".format(me.get_battery()))
        break


#ff.close()

