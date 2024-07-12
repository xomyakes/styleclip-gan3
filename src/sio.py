import logging
import socketio
from fastapi import FastAPI

sio = socketio.AsyncServer(cors_allowed_origins=["*"],async_mode='asgi')
logger = logging.getLogger(__name__)

def init_sio(app: FastAPI):
    
    app.mount('/ws', socketio.ASGIApp(sio,socketio_path="/ws/socket.io"))
    logger.info(f"SIO initialized")