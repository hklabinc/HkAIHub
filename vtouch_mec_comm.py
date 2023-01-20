import asyncio
import websockets
import time

URL = 'ws://localhost:20000'

class VTouchMecComm:
    def __init__(self):
        self.ws = None
        self.loop = asyncio.get_event_loop()
        self.loop.run_until_complete(self.__async__connect())

    async def __async__connect(self):
        self.ws = await websockets.connect(URL)
        
    def send(self, message):
        return self.loop.run_until_complete(self.__async__send(message))
    
    async def __async__send(self, message):
        await self.ws.send(message)

# comm = VTouchMecComm()

# while True:
#     comm.send('hello')
#     time.sleep(1)