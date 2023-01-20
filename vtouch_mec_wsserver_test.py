import asyncio
import websockets
import json
import cv2
import base64
import numpy as np

from websockets import WebSocketServerProtocol


# Plots one bounding box on image img
def plot_one_box(x, img, color=[0, 255, 0], label=None, line_thickness=2):
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    height = img.shape[0]
    width = img.shape[1]    
    c1, c2 = (int(x[0]*width), int(x[1]*height)), (int(x[2]*width), int(x[3]*height))    
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)    
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


class WSServer:
    clients = set()

    async def register(self, ws: WebSocketServerProtocol) -> None:
        self.clients.add(ws)
        print(f'{ws.remote_address} connects.')

    async def unregister(self, ws: WebSocketServerProtocol) -> None:
        self.clients.remove(ws)
        print(f'{ws.remote_address} disconnects.')

    async def ws_handler(self, ws: WebSocketServerProtocol, uri: str) -> None:
        await self.register(ws)
        try:
            await self.on_read(ws)
        finally:
            await self.unregister(ws)

    async def on_read(self, ws: WebSocketServerProtocol) -> None:
        async for message in ws:        
            json_dict = json.loads(message)

            string_image = json_dict['image']
            json_dict.pop('image', None)

            bytes_image = bytes(string_image, 'utf-8')
            jpg_image = base64.b64decode(bytes_image)
            np_image = np.frombuffer(jpg_image, dtype=np.uint8)  # im_arr is one-dim Numpy array
            img = cv2.imdecode(np_image, flags=cv2.IMREAD_COLOR)

       
            ## Add box to image ######################################
            jsonArray = json_dict.get("detection_boxes")
            for list in jsonArray:
                xywh = (list.get("x_center"), list.get("y_center"), list.get("width"), list.get("height"))
                conf = list.get("confidence")
                label = list.get("label")

                xyxy = (xywh[0] - xywh[2] / 2,  # top left x
                        xywh[1] - xywh[3] / 2,  # top left y
                        xywh[0] + xywh[2] / 2,  # bottom right x
                        xywh[1] + xywh[3] / 2)  # bottom right y

                # Plots one bounding box on image img
                plot_one_box(xyxy, img, color=[0, 255, 0], label=f'{label} {conf:.2f}', line_thickness=1)
            ##########################################################


            cv2.imshow(f'{ws.remote_address}', img)
            # cv2.imshow(f'{ws.remote_address}', cv2.resize(img, dsize=(640, 480), interpolation=cv2.INTER_AREA))
            cv2.waitKey(1)

            print(f'{ws.remote_address}' + ": \n" + json.dumps(json_dict) + "\n")
            #print(f'{ws.remote_address}' + ": " + message)


server = WSServer()
start_server = websockets.serve(server.ws_handler, 'localhost', 20000)

loop = asyncio.get_event_loop()
loop.run_until_complete(start_server)
loop.run_forever()