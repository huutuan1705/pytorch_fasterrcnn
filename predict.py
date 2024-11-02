import torch 
import cv2
import os 
import numpy as np
import base64
from io import BytesIO

def rgb_to_base64(image):
    buffer = BytesIO()
    image.save(buffer, format="jpg")
    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return img_str

def base64_to_rgb(base64_str):
    img_data = base64.b64decode(base64_str)
    np_arr = np.frombuffer(img_data, np.uint8) 
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return image

def predict(image_base64, model, device, score_threshold=0.6):
    # image = base64_to_rgb(image_base64)
    image = image_base64
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_tensor = torch.tensor(image_rgb).permute(2, 0, 1).float().to(device) / 255.0
    image_tensor = image_tensor.unsqueeze(0)  
    
    model.eval()
    with torch.no_grad():
        predictions = model(image_tensor)
    
    boxes = predictions[1]['boxes'].cpu().numpy()
    scores = predictions[1]['scores'].cpu().numpy()
    labels = predictions[1]['labels'].cpu().numpy()

    list_label = ["mild", "moderate", "severe"]
    for i, box in enumerate(boxes):
        if scores[i] >= score_threshold:
            x_min, y_min, x_max, y_max = box.astype(int)
            label = labels[i]
            score = scores[i]
            print(x_min, y_min, x_max, y_max, label, score)
            cv2.rectangle(image_rgb, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(image_rgb, f'{list_label[int(label)-1]}: {score:.2f}', 
                        (x_min-20, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    output_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    # return rgb_to_base64(output_image)
    return output_image
    
def visualize_predictions(image_path, predictions, score_threshold=0.6, output_folder = 'outputs'):
    image_name = os.path.basename(image_path)
    output_path = os.path.join(output_folder, image_name)
    
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    boxes = predictions[1]['boxes'].cpu().numpy()
    scores = predictions[1]['scores'].cpu().numpy()
    labels = predictions[1]['labels'].cpu().numpy()
    
    list_label = ["mild", "moderate", "severe"]
    for i, box in enumerate(boxes):
        if scores[i] >= score_threshold:
            x_min, y_min, x_max, y_max = box.astype(int)
            label = labels[i]
            score = scores[i]
            print(x_min, y_min, x_max, y_max, label, score)
            cv2.rectangle(image_rgb, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(image_rgb, f'{list_label[int(label)-1]}: {score:.2f}', 
                        (x_min-20, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    output_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR) 
    cv2.imwrite(output_path, output_image)        
            
# image_path = "images/image_2874.jpg"
# predictions = predict(image_path, faster_rcnn_model)
# visualize_predictions(image_path, predictions)