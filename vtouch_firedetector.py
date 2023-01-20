import torch
import numpy as np

from models.experimental import attempt_load, attempt_load_v5
from utils.datasets import letterbox
from utils.general import check_img_size, check_img_size_v5, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, set_logging
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized, TracedModel

from vtouch_mec_ai_data import DetectionBox, VTouchLabel

class VTouchFireDetector:
    def __init__(self, weights, weights_c, classify):    
        img_size = 640
        trace = True                # trace model
        self.iou_thres = 0.45       # IOU threshold for NMS
        self.classes = None         # filter by class: --class 0, or --class 0 2 3
        self.agnostic_nms = False   # class-agnostic NMS
        self.augment = True         # augmented inference
        self.device = ''            # '' or 'cpu'
        self.classify = classify    # Second-stage classifier  

        # Initialize
        set_logging()
        self.device = select_device(self.device)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        self.model = attempt_load(weights, map_location=self.device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(img_size, s=self.stride)  # check img_size

        if trace:
            self.model = TracedModel(self.model, self.device, img_size)

        if self.half:
            self.model.half()  # to FP16

        # Get names and colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in self.names]
        
        # Run inference
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once
        self.old_img_w = self.old_img_h = self.imgsz
        self.old_img_b = 1

        # Second-stage classifier        
        if self.classify:
            # Load model
            modelc = attempt_load_v5(weights_c, device=self.device, inplace=True, fuse=True)
            stridec = max(int(modelc.stride.max()), 32)  # model stride
            self.namesc = modelc.module.names if hasattr(modelc, 'module') else modelc.names  # get class names            
            modelc.float()
            self.modelc = modelc
            self.imgszc = check_img_size_v5((224, 224), s=stridec)  # check image size            

            # warmup  
            fp16 = False            
            im2 = torch.empty(*(1, 3, *self.imgszc), dtype=torch.half if fp16 else torch.float, device=self.device)  # input                
            modelc(im2)


    def detect(self, img0, conf_thres, draw_box):     
        # LoadImages        
        img = letterbox(img0, self.imgsz, stride=self.stride)[0]    # Padded resize
        img = img[:, :, ::-1].transpose(2, 0, 1)  # Convert BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)                 
        
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if self.device.type != 'cpu' and (self.old_img_b != img.shape[0] or self.old_img_h != img.shape[2] or self.old_img_w != img.shape[3]):
            self.old_img_b = img.shape[0]
            self.old_img_h = img.shape[2]
            self.old_img_w = img.shape[3]
            for i in range(3):
                self.model(img, self.augment)[0]

        # Inference
        t1 = time_synchronized()
        pred = self.model(img, self.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, self.iou_thres, self.classes, self.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if self.classify:            
            pred = apply_classifier(pred, self.modelc, img, img0, self.imgszc, self.namesc, self.device)            
        
        # Process detections
        result = []
        for i, det in enumerate(pred):  # detections per image           

            s = ''
            gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                        
            if len(det):
                # Rescale boxes from img_size to img0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Add box to image
                for *xyxy, conf, cls in reversed(det):
                    if draw_box:
                        label = f'{self.names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, img0, label=label, color=self.colors[int(cls)], line_thickness=2)
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    #line = (self.names[int(cls)], *xywh, conf) # label format                    

                    result.append(DetectionBox(*xywh, conf.item(), VTouchLabel(int(cls)+1)))     # int(cls)==0 is fire, int(cls)==1 is smoke
                
        # Print time (inference + NMS)
        print(f'Detected: {s} Inference ({(1E3 * (t2 - t1)):.1f}ms), NMS ({(1E3 * (t3 - t2)):.1f}ms)')

        return result, img0


