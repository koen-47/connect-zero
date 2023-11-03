import requests

data = {"player": 1,
        "board": [[0,  0, 0, 0, 0, 0, 0],
                  [0,  0, 0, 0, 0, 0, 0],
                  [0,  0, 0, 0, 0, 0, 0],
                  [0, -1, 0, 0, 0, 0, 0],
                  [0, -1, 1, 0, 0, 0, 0],
                  [0, -1, 1, 1, 0, 0, 0]]}

response = requests.post("http://localhost:5000/connect-zero/predict", json=data)
print("Response:", response.json())
