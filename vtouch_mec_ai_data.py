from email.mime import base
from enum import Enum
from typing import List
from datetime import datetime

import json

class CameraId(Enum):
    AIRSTRIP_1 = 0
    AIRSTRIP_2 = 1
    AIRSTRIP_3 = 2
    AIRSTRIP_4 = 3
    VEHICLE_1 = 4
    VEHICLE_2 = 5
    GUNPOWDER_HOUSE = 6

class VTouchLabel(Enum):
    BIRD = 0
    FIRE = 1
    SMOKE = 2
    PERSON = 3

class DetectionBox:
    def __init__(self, x_center: float, y_center: float, width: float, height: float, confidence: float, label: VTouchLabel):
        self.x_center = x_center
        self.y_center = y_center
        self.width = width
        self.height = height
        self.confidence = confidence
        self.label = label.name

    def toJson(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)

class VTouchMecAiData:
    def __init__(self, camera_id: CameraId, base64encoded_image: str, detection_boxes:List[DetectionBox]):
        self.time = datetime.now().strftime('%y-%m-%d %H:%M:%S')
        self.camera_id = camera_id.name
        self.image = base64encoded_image
        self.detection_boxes = detection_boxes

    def toJson(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)

# test = VTouchMecAiData(CameraId.GUNPOWDER_HOUSE, "test", AlgorithmType.FIRE, [DetectionBox(0.2, 0.2, 0.1, 0.1, 0.9)])
# print (test.toJson())