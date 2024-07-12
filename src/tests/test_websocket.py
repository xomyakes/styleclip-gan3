import asyncio
import socketio
from test_mapper import send_request as test_mapper_request
from test_optimization import send_request as test_optimization_request

URL = "http://localhost:8000"
sio = socketio.AsyncClient()

@sio.event
async def connect():
    print('Connection established')

@sio.event
async def disconnect():
    print('Disconnected from server')

@sio.event
async def image_edited(data):
    print('Image edited:', data)

@sio.event
async def image_processing(data):
    print(f"Editing progress: {data.get('step')}/{data.get('steps')}")

async def main():
    await sio.connect(URL, socketio_path="/ws/socket.io")
    print(sio.get_sid())
    test_optimization_request(sid = sio.get_sid())
     # test_mapper_request(sio.get_sid())    
    print("Request for editing done")
    await sio.wait()

if __name__ == "__main__":
    asyncio.run(main())