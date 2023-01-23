# HkAIHub
2023 HkAIHub

main_AI_hub.py로 실행
0.1초 정도 간격으로 Object detection -> Image classification 두 단계 동작 

- 환경
  - Object detection은 yolov7, Image classification은 yolov5를 기반
  - 윈도우 환경, Python 3.9.12에서 개발
  - pip install -r requirements.txt 먼저 실행 필요
  - Linux Ubuntu에서 동작 확인
  - Docker (python:3.9-slim) 환경에서 동작 확인

- 실행
  1) git clone https://github.com/hklabinc/HkAIHub.git
  2) pip install -r requirements.txt
  3) python3 main_AI_hub.py

- 파일
  - main_AI_hub : fire detector 실행
  - vtouch_firedetector : 메인 fire detector 동작
  - Yolov7 및 v5 소스코드들이 data, models, utils 폴더에 있음 (몇몇 파일 수정됨)
  - weights 폴더에 개발한 od와 ic 모델 pt 파일이 있음
     - od_fire_smoke.pt : yolov7 기반 obeject detection (fire, smoke 검출)
     - ic_default_fire_smoke.pt: yolov5m 기반 image classification(0: default, 1: fire, 2: smoke)
  
- 옵션 (test_firedetector에서)
  - IS_CLASSIFY : 두번째 IC 사용 여부 선택 (True일 경우 image classification 추가 사용, False일 경우 object detection만 사용)
  - IS_SMALL_YOLOv7_OD : YOLOv7 Object Detection의 Small 모델 사용 여부 (True일 경우 Small 모델 사용, False일 경우 Medium 모델 사용)
  - IS_SMALL_YOLOv5_IC : YOLOv5 Image Classification의 Small 모델 사용 여부 (True일 경우 Small 모델 사용, False일 경우 Medium 모델 사용)
  - CONFIDENCE_THRESHOLD : 오브젝트 검출을 위한 confidence 값 (default=0.25)
  - DETECT_PERIOD : 매 DETECT_PERIOD 마다 캡쳐된 이미지가 큐에 들어감 (default=0.1)             
  - MAX_QUEUE_SIZE : 최대 큐 사이즈로 이 값에 다다르면 큐가 비워짐 (default=10)   
  - draw_box : 이미지에 검출된 오브젝트의 box 그리기 여부 선택
