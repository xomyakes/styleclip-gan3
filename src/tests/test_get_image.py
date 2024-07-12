import os
import requests
import random
import shutil

URL = "http://localhost:8000"
i = random.randint(0,29999)
EDITED_IMAGE_PATH = "./images/styled/62.jpg"

def send_request():
    response = requests.get(f"{URL}/image/{EDITED_IMAGE_PATH}",stream=True)
    if response.status_code == 200:
        with open(os.path.basename(EDITED_IMAGE_PATH),"wb") as file:
            response.raw.decode_content = True
            shutil.copyfileobj(response.raw,file)
    del response

if __name__ == "__main__":
    send_request()