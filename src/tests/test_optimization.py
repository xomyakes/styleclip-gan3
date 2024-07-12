import requests
import random

URL = "http://localhost:8000"
i = random.randint(0,29999)
# IMAGE_TO_EDIT = f"./dataset/CelebAMask-HQ/CelebA-HQ-img/{i}.jpg"
IMAGE_TO_EDIT = f"./dataset/CelebAMask-HQ/CelebA-HQ-img/1.jpg"

def send_request(prompt: str = "sad", sid: str = "1"):
    params = {
        "sid" : sid,
        "prompt" : prompt
    }
    response = requests.post(f"{URL}/edit",params=params,files={"image" : open(IMAGE_TO_EDIT,"rb")})
    print(response.status_code)
    print(response.json())

if __name__ == "__main__":
    send_request()