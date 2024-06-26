
from flask import Flask, render_template, request
from ultralytics import YOLO
import uuid
import os
import cv2
import yaml
import string
from datetime import datetime

app = Flask(__name__)
model_plate = YOLO("./model/platedetect.pt")
model_number = YOLO("./model/numberdetect.pt")

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

alp = {i+10: list(string.ascii_uppercase)[i] for i in range(26)}
for i in range(10):
    alp[i] = i
alp = {str(key): str(value) for key, value in alp.items()}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    imagefile= request.files['imagefile']
    if imagefile and allowed_file(imagefile.filename):
        ext = os.path.splitext(imagefile.filename)[1]

        # Generate the timestamp filename
        timestamp = datetime.now().strftime("%Y-%b-%d_%H-%M-%S")
        filename = f"{timestamp}{ext}"

        image_path = "./images/" + filename
        imagefile.save(image_path)

        image = cv2.imread(image_path)
        results = model_plate(image)

        # Iterate over the results, extract bounding box coordinates for each detected license plate,
        # crop the license plate from the original image, and save the cropped image
        for idx, result in enumerate(results):
            for box in result.boxes.xyxy:
                x1, y1, x2, y2 = map(int, box[:4]) 
                cropped_image = image[y1:y2, x1:x2]
                crop_output_path = os.path.join("cropped_images", f"crop_{idx}.png")
                cv2.imwrite(crop_output_path, cropped_image)

        with open('data.yaml', 'r') as file:
            config = yaml.safe_load(file)

        class_names = config['names']

        crop_img_path = 'cropped_images/crop_0.png'
        results = model_number(crop_img_path)

        # Iterate over the results, extract bounding box coordinates and class IDs for each detected character,
        # convert class IDs to characters, sort the characters by their x-coordinate, and join the characters together into a string
        lines = []
        for result in results:
            boxes = result.boxes
            line = []
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                class_id = int(box.cls[0].item())
                character = alp[class_names[class_id]]
                line.append((character, x1, y1, x2, y2, conf))
            line.sort(key=lambda x: x[1])
            line = " ".join([char for char, x1, y1, x2, y2, conf in line])
            lines.append(line)

        return render_template('index.html', prediction=line)
    else:
        return render_template('index.html', error="Invalid file type. Please upload an image file.")

if __name__ == '__main__':
    app.run(port=3000, debug=True)