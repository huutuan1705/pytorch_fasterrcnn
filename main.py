import torch

import yaml
import argparse
import uvicorn
import cv2
import io
import numpy as np
import gdown
import os
from pydantic import BaseModel

from faster_rcnn import FasterRCNN
from predict import predict

from fastapi import FastAPI, UploadFile
from fastapi.responses import StreamingResponse

config_path = 'decay.yaml'
args = argparse.Namespace(config_path=config_path)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open(args.config_path, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

if not os.path.exists("faster_rcnn_decay.pth"):
    file_id = '1FFq1yXeeOUZTumhhuxsZH-c7xQXzlZHH'
    gdown.download(f'https://drive.google.com/uc?id={file_id}', 'faster_rcnn_decay.pth', quiet=False)
    
model_config = config['model_params']
train_config = config['train_params']           
faster_rcnn_model = FasterRCNN(model_config, num_classes=4)
checkpoint_path = "faster_rcnn_decay.pth"

faster_rcnn_model.eval()
faster_rcnn_model.to(device)
faster_rcnn_model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device(device)))

app = FastAPI()

class Input(BaseModel):
    img_base64: str

@app.get("/")
async def root():
    return {
        "message": "Hello world"
    }


@app.post("/api/predict")
async def predict_api(file: UploadFile):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    output_img = predict(image, faster_rcnn_model, 'cpu')
    
    _, buffer = cv2.imencode('.jpg', output_img)
    io_buf = io.BytesIO(buffer)
    
    return StreamingResponse(io_buf, media_type="image/png")
    
if __name__ == "__main__":
    uvicorn.run("main:app", host='localhost', port=8080, reload=True) 
    