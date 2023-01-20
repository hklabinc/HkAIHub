from vtouch_mec_ai_data import CameraId, DetectionBox, VTouchLabel, VTouchMecAiData
from vtouch_mec_comm import VTouchMecComm
from vtouch_firedetector import VTouchFireDetector

import cv2, torch
import platform, pathlib
import sys, base64, time, queue, threading


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
url = 'rtsp://admin:init123!!@1.237.139.6:554/SD'
# url = 'rtsp://admin:init123!!@192.168.0.59:554/HD'
# url = 'rtsp://sonslab:sons123!@hklab-cam02.iptimecam.com:21064/stream_ch00_0'
# url = 'rtsp://admin:tech0316_@218.145.166.65:554/MOBILE'    # Vtouch Camera
# url = 'datasets/ONO-9081R_20221024164811.avi'               # Pyeongtak
# url = 'rtsp://'
url = 0

print('\033[95m' + "Connect to server..." + '\033[0m')
comm = VTouchMecComm()

print('\033[95m' + "Initialize Yolo..." + '\033[0m')
fd = VTouchFireDetector(weights, weights_c, classify=IS_CLASSIFY)      # Set classify=True if want to use second-stage classification


### Receiving Thread ###
def Receive():
    print('\033[95m' + "Start Reveive thread..." + '\033[0m')    
    past = time.time()
    cap = cv2.VideoCapture(url)
    while True :
        ret, frame = cap.read()
        cv2.waitKey(1)

        if not(ret):                                # If RTSP stream is lost, reinitialize
            st = time.time()
            cap = cv2.VideoCapture(url)                 
            print('\033[95m' + f'RTSP stream is reinitialized, lost time is {time.time()-st}...' + '\033[0m')    
        else:            
            if q.qsize() > MAX_QUEUE_SIZE:          # Prevent queue overflow
                print('\033[95m' + f'Current queue size of {q.qsize()} is too long, drop frames...' + '\033[0m')    
                q.queue.clear()

            now = time.time()        
            if now - past >= DETECT_PERIOD:         # for each period
                past = now
                q.put(frame)

### Processing Thread ### 
def Process():
    print('\033[95m' + "Start Process thread..." + '\033[0m')    
    
    while True:      
        cv2.waitKey(1)

        if q.empty() != True:       
            frame_image = q.get()  

            with torch.no_grad():       
                result, frame_det = fd.detect(frame_image, conf_thres=CONFIDENCE_THRESHOLD, draw_box=True)     # Inference with Yolo
                
            # cv2.imshow("Video_detected", frame_det)       # Show the image with detections
        
            if len(result) > 0:     # Only if anything is detected, send to server
                ret, jpg_image = cv2.imencode('.jpg', frame_det)
                base64_image = base64.b64encode(jpg_image)
                data = VTouchMecAiData(CameraId.GUNPOWDER_HOUSE, str(base64_image, 'utf-8'), result)
                comm.send(data.toJson())


### Main Thread ###
if __name__=='__main__':
    try:
        p1 = threading.Thread(target=Receive, daemon=True)       # https://stackoverflow.com/questions/49233433/opencv-read-errorh264-0x8f915e0-error-while-decoding-mb-53-20-bytestream 
        p2 = threading.Thread(target=Process, daemon=True)        
        p1.start()
        p2.start()
        
        while True:
            time.sleep(100)
    except (KeyboardInterrupt, SystemExit):
        print('\033[95m' + 'Keyboard interrupted, Quitting program.\n' + '\033[0m') 
        sys.exit()
