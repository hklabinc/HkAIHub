# from vtouch_mec_ai_data import CameraId, DetectionBox, VTouchLabel, VTouchMecAiData
# from vtouch_mec_comm import VTouchMecComm
from vtouch_firedetector import VTouchFireDetector

import cv2, torch
import platform, pathlib
import sys, base64, time, queue, threading
import paho.mqtt.client as mqtt     # For MQTT
import json
from PIL import Image
from io import BytesIO
import datetime
import numpy as np
import matplotlib.pyplot as plt


### Path setup in case of Linux
if platform.system() == 'Linux':        
    print('\033[95m' + "Set path for linux..." + '\033[0m')
    pathlib.WindowsPath = pathlib.PosixPath     

### Initialization ###
IS_CLASSIFY = True              # True if want to use second-stage image classification
IS_SMALL_YOLOv7_OD = True      # True if want the small model for YOLOv7 Object Detection 
IS_SMALL_YOLOv5_IC = True       # True if want the small model for YOLOv5 Image Classification
CONFIDENCE_THRESHOLD = 0.25     # If the confidence value is less than CONFIDENCE_THRESHOLD, the object is detected
DETECT_PERIOD = 0.1             # Captured image is put into the queue every DETECT_PERIOD
MAX_QUEUE_SIZE = 10             # If queue size is greater than MAX_QUEUE_SIZE, queue becomes clear
q = queue.Queue()
weights   = 'weights/od_small_fire_smoke.pt'         if IS_SMALL_YOLOv7_OD else 'weights/od_medium_fire_smoke.pt'   
weights_c = 'weights/ic_small_default_fire_smoke.pt' if IS_SMALL_YOLOv5_IC else 'weights/ic_medium_default_fire_smoke.pt'   

# url = 'rtsp://admin:init123!!@192.168.0.59:554/SD'
# url = 'rtsp://admin:init123!!@sean715.iptime.org:554/SD'
# url = 'rtsp://admin:init123!!@1.237.139.6:554/SD'
# url = 'rtsp://admin:init123!!@192.168.0.59:554/HD'
# url = 'rtsp://sonslab:sons123!@hklab-cam02.iptimecam.com:21064/stream_ch00_0'
# url = 'rtsp://admin:tech0316_@218.145.166.65:554/MOBILE'    # Vtouch Camera
# url = 'datasets/ONO-9081R_20221024164811.avi'               # Pyeongtak
# url = 'rtsp://'
# url = 0

# print('\033[95m' + "Connect to server..." + '\033[0m')
# comm = VTouchMecComm()

print('\033[95m' + "Initialize Yolo..." + '\033[0m')
fd = VTouchFireDetector(weights, weights_c, classify=IS_CLASSIFY)      # Set classify=True if want to use second-stage classification



## For MQTT #########################################################################################################
# ?????? ?????? ????????????
#  (mqttc.connect??? ??? ??????) ?????? ????????? ????????? on_connect ?????? (???????????? ???????????? ??????)
def on_connect(client, userdata, flags, rc):
    print("rc: " + str(rc))
   
# (mqttc.subscribe??? ??? ??????) ??????(subscribe)??? ????????????
# on_subscrbie??? ????????? (???????????? ???????????? ?????????)
def on_subscribe(client, obj, mid, granted_qos):
    print("Subscribe complete : " + str(client)+ ", "  + str(mid) + " " + str(granted_qos))

# ??????????????? ???????????? ???????????? on_message ?????? (???????????? ???????????? ??????)
def on_message(client, obj, msg):
    print(msg.topic + ", " + str(client)+ ", " + str(msg.qos) + ", " + str(msg.payload)[0:100])

    # payload = msg.payload.decode('utf8')
    payload = msg.payload
    # json_obj = json.loads(payload)
    # addr = json_obj["addr"]
    # time = json_obj["time"]
    # type = json_obj["type"]
    # label = json_obj["label"]
    # image = json_obj["image"]    
    # print("Received MQTT msg: ", addr, time, type, label, image[0:50])


    # global isImage, isEvent          # global??? ????????????    
    # global para_interval, para_scale, para_width, para_height

    # command = msg.payload.decode('utf8')
    # if "isImage" in command:       
    #     isImage = to_bool(command.split("=")[1])
    # elif "isEvent" in command:  
    #     isEvent = to_bool(command.split("=")[1])
    # elif "interval" in command:
    #     para_interval = float(command.split("=")[1])
    # elif "scale" in command:
    #     para_scale = float(command.split("=")[1])        

    
                   
    if q.qsize() > MAX_QUEUE_SIZE:          # Prevent queue overflow
        print('\033[95m' + f'Current queue size of {q.qsize()} is too long, drop frames...' + '\033[0m')    
        q.queue.clear()
    q.put(payload)      # Insert the payload to queue
                
       
 
# (mqttc.publish??? ??? ??????) ???????????? publish?????? on_publish?????? (???????????? ???????????? ??????)
def on_publish(client, obj, mid):
    # ?????? : publish??? ????????? ??? ??? ????????? ?????? ?????? ???
    # ?????? ??? ??????????????? ??? ?????? ?????????.
    print("mid: " + str(mid) + ", " + str(client))
 
# ??????????????? ??????
mqttc = mqtt.Client()
#mqttc = mqtt.Client("Cam_01")

# ?????? ?????? ????????????
mqttc.on_message = on_message
mqttc.on_connect = on_connect
# mqttc.on_publish = on_publish
mqttc.on_subscribe = on_subscribe

# ????????? ?????? ??????
# ????????? ???????????? Cloudmqtt ??????????????? ????????? ??????
# ?????? username??? password, topic??? ?????? ??????????????????.
url = "hawkai.hknu.ac.kr"
port = 8085
pub_topic = "hawkai/from"  
sub_topic = "hawkai/query"
#username = "HONG" # Cloud mqtt
#password = "1234"
  
# ??????????????? ?????? ??? ?????? ??????
#mqttc.username_pw_set(username, password)
mqttc.connect(host=url, port=port)
#mqttc.connect("ictrobot.hknu.ac.kr", 8085)
 
# QoS level 0?????? ?????? ??????, ??????????????? subscribe ?????? on_subscribe ?????????
mqttc.subscribe(sub_topic, 0)
mqttc.loop_start()
#########################################################################################################



# ### Receiving Thread ###
# def Receive():
#     print('\033[95m' + "Start Reveive thread..." + '\033[0m')    
#     past = time.time()
#     cap = cv2.VideoCapture(url)
#     while True :
#         ret, frame = cap.read()
#         cv2.waitKey(1)

#         if not(ret):                                # If RTSP stream is lost, reinitialize
#             st = time.time()
#             cap = cv2.VideoCapture(url)                 
#             print('\033[95m' + f'RTSP stream is reinitialized, lost time is {time.time()-st}...' + '\033[0m')    
#         else:            
#             if q.qsize() > MAX_QUEUE_SIZE:          # Prevent queue overflow
#                 print('\033[95m' + f'Current queue size of {q.qsize()} is too long, drop frames...' + '\033[0m')    
#                 q.queue.clear()

#             now = time.time()        
#             if now - past >= DETECT_PERIOD:         # for each period
#                 past = now
#                 q.put(frame)

### Processing Thread ### 
def Process():
    print('\033[95m' + "Start Process thread..." + '\033[0m')    
    
    while True:      
        cv2.waitKey(1)

        if q.empty() != True:                  
            payload = q.get()  
            json_obj = json.loads(payload)
            addr = json_obj["addr"]
            timeValue = json_obj["time"]
            userId = json_obj["type"]
            # label = json_obj["label"]
            image = json_obj["image"]    
            decoded_data = base64.b64decode(image)        # decode base64 string data
            frame = Image.open(BytesIO(decoded_data))   # Load image from BytesIO    (BytesIO(decoded_data)??? jpg ????????? ??????)            
            frame_image = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)  # Convert to numpy array, then covert to BGR (opencv format) -> ?????? detect??? ???????????? BGR??? ???????????????
                        
            # ????????? ?????? ????????? ???????????? ??????????????? ??????????????? ?????? ?????? ?????? ??? ?????? 3??? ?????? ????????? ??? ?????? (??? 3ms)
            # frame.save("image.jpg")
            # frame_image = cv2.imread("image.jpg")  # BGR              

            with torch.no_grad():       
                result, frame_det = fd.detect(frame_image, conf_thres=CONFIDENCE_THRESHOLD, draw_box=True)     # Inference with Yolo
                
            # cv2.imshow("Video_detected", frame_det)       # Show the image with detections
        
            if len(result) > 0:     # Only if anything is detected, send to server
                ret, jpg_image = cv2.imencode('.jpg', frame_det)
                base64_image = base64.b64encode(jpg_image).decode('utf8')      # Convert to base64 string  
                
                json_object = {
                    "addr": addr,
                    "time": timeValue,
                    "type": "event",                
                    "label": "fire",     # TBD          
                    "image": base64_image
                    }
                
                mqttc.publish(pub_topic+"/"+userId, json.dumps(json_object))
                print(f"[HHCHOI] Sent Event of size {sys.getsizeof(json.dumps(json_object))} at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")


### Main Thread ###
if __name__=='__main__':
    try:
        # p1 = threading.Thread(target=Receive, daemon=True)       # https://stackoverflow.com/questions/49233433/opencv-read-errorh264-0x8f915e0-error-while-decoding-mb-53-20-bytestream 
        p2 = threading.Thread(target=Process, daemon=True)        
        # p1.start()
        p2.start()
        
        while True:
            time.sleep(100)
    except (KeyboardInterrupt, SystemExit):
        print('\033[95m' + 'Keyboard interrupted, Quitting program.\n' + '\033[0m') 
        sys.exit()
