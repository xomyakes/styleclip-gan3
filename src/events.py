import logging
from .sio import sio

logger = logging.getLogger(__name__)

@sio.event
async def connect(sid, environ):
    logger.info(f"Client connected: {sid}")
    return sid

@sio.event
async def disconnect(sid):
    logger.info(f"Client disconnected: {sid}")
    return 