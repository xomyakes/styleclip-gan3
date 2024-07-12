import requests

URL = "http://localhost:8000"


def get_mappers():
    response = requests.get(f"{URL}/mappers")
    print(response.status_code)
    print(response.json())

if __name__ == "__main__":
    get_mappers()