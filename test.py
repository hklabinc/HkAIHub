import cv2
import base64

from vtouch_mec_ai_data import CameraId, DetectionBox, AlgorithmType, VTouchMecAiData

from vtouch_mec_comm import VTouchMecComm

# 샘플 영상
url = 'rtsp://admin:init123!!@sean715.iptime.org:554/SD'
cap = cv2.VideoCapture(url)

comm = VTouchMecComm()

while True :
    ret, frame_iamge = cap.read()
    ret, jpg_image = cv2.imencode('.jpg', frame_iamge)
    base64_image = base64.b64encode(jpg_image)

    data = VTouchMecAiData(CameraId.GUNPOWDER_HOUSE, str(base64_image, 'utf-8'), AlgorithmType.FIRE, [DetectionBox(0.2, 0.2, 0.1, 0.1, 0.9)])
    comm.send(data.toJson())