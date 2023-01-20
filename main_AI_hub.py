# from vtouch_mec_ai_data import CameraId, DetectionBox, VTouchLabel, VTouchMecAiData
# from vtouch_mec_comm import VTouchMecComm
from vtouch_firedetector import VTouchFireDetector

import cv2, torch
import platform, pathlib
import sys, base64, time, queue, threading, json
import paho.mqtt.client as mqtt     # For MQTT
from PIL import Image
from io import BytesIO
import numpy as np



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
# 콜백 함수 정의하기
#  (mqttc.connect를 잘 되면) 서버 연결이 잘되면 on_connect 실행 (이벤트가 발생하면 호출)
def on_connect(client, userdata, flags, rc):
    print("rc: " + str(rc))
   
# (mqttc.subscribe가 잘 되면) 구독(subscribe)을 완료하면
# on_subscrbie가 호출됨 (이벤트가 발생하면 호출됨)
def on_subscribe(client, obj, mid, granted_qos):
    print("Subscribe complete : " + str(client)+ ", "  + str(mid) + " " + str(granted_qos))

# 브로커에게 메시지가 도착하면 on_message 실행 (이벤트가 발생하면 호출)
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


    # global isImage, isEvent          # global로 선언해야    
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
                
       
 
# (mqttc.publish가 잘 되면) 메시지를 publish하면 on_publish실행 (이벤트가 발생하면 호출)
def on_publish(client, obj, mid):
    # 용도 : publish를 보내고 난 후 처리를 하고 싶을 때
    # 사실 이 콜백함수는 잘 쓰진 않는다.
    print("mid: " + str(mid) + ", " + str(client))
 
# 클라이언트 생성
mqttc = mqtt.Client()
#mqttc = mqtt.Client("Cam_01")

# 콜백 함수 할당하기
mqttc.on_message = on_message
mqttc.on_connect = on_connect
# mqttc.on_publish = on_publish
mqttc.on_subscribe = on_subscribe

# 브로커 연결 설정
# 참고로 브로커를 Cloudmqtt 홈페이지를 사용할 경우
# 미리 username과 password, topic이 등록 되어있어야함.
url = "hawkai.hknu.ac.kr"
port = 8085
pub_topic = "hawkai/from"  
sub_topic = "hawkai/query"
#username = "HONG" # Cloud mqtt
#password = "1234"
  
# 클라이언트 설정 후 연결 시도
#mqttc.username_pw_set(username, password)
mqttc.connect(host=url, port=port)
#mqttc.connect("ictrobot.hknu.ac.kr", 8085)
 
# QoS level 0으로 구독 설정, 정상적으로 subscribe 되면 on_subscribe 호출됨
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
            decoded_data = base64.b64decode(image)          # Decode base64 string data
            frame = Image.open(BytesIO(decoded_data))       # Load image from BytesIO    (BytesIO(decoded_data)는 jpg 파일에 해당)            
            frame_image = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)  # Convert to numpy array, then covert to BGR (opencv format) -> 아래 detect에 들어갈때 BGR로 들어가야함
                        
            # 아래와 같이 파일을 저장하고 불러들여도 동작하지만 위와 같이 하는 것 대비 3배 이상 시간이 더 걸림 (약 3ms)
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
                print(f"[HHCHOI] Sent Event of size {sys.getsizeof(json.dumps(json_object))}")


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
