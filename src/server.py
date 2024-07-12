from contextlib import asynccontextmanager
import logging
import os
import traceback
from typing import Annotated
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, UploadFile, File, status
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from .sio import init_sio
from .logger_config import configure_logging
from glob import glob


configure_logging()

logger = logging.getLogger(__name__)
ip = None

async def exception_handler(request, call_next):
    try:
        return await call_next(request)
    except Exception as exc:
        exc_type, exc_value, exc_tb = exc.__class__, exc, exc.__traceback__
        error_message = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        logger.exception(error_message)
        return JSONResponse({"err": str(error_message)}, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)


@asynccontextmanager
async def lifespan(application: FastAPI):
    logger.info("Starting")
    from . import events 
    from .sio import sio
    from .image_processor import ImageProcessor
    global ip
    ip = ImageProcessor(sio)
    await ip.start()
    yield
    await ip.stop()
    logger.info("Shutting down")

def init_app() -> FastAPI:
    _app = FastAPI(lifespan=lifespan, title="Challenge")
    init_sio(_app)
    _app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    _app.middleware("http")(exception_handler)
    return _app

app = init_app()

@app.get("/mappers")
async def get_pretrained_mappers_list():
    mapper_files = glob("./mappers/*.pt")
    mappers = [os.path.basename(mapper_file[:mapper_file.index(".pt")]) for mapper_file in mapper_files]
    return mappers

@app.post("/edit")
async def edit_image(
    image: Annotated[UploadFile,File()],
    sid: Annotated[str,Query()],
    prompt: Annotated[str,Query()] = None,
    mapper: Annotated[str,Query()] = None,
):
    if not sid:
        raise HTTPException(f"Please send your sid from ws connection to get a notification of completing edition")
    if not prompt and not mapper:
        raise HTTPException(400, f"Please choose pre-trained mapper or write a prompt for editing through vector optimization")
    file_location  = f"images/real/{os.path.basename(image.filename)}"
    logger.info(file_location)
    with open(file_location, "wb") as file:
        file.write(image.file.read())
    styled_path, restored_path = await ip.add_image_to_queue(sid, file_location, prompt, mapper)
    return JSONResponse({'image_path': styled_path, "restored_path" : restored_path})
    
    
@app.get("/image/{image_path:path}")
async def get_edited_image(image_path:str):
    return FileResponse(image_path)