from flask import Flask, request, jsonify, send_from_directory
from PIL import Image, ImageDraw, ImageFont
import torch
import os
import uuid
import cv2
import numpy as np
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from transformers import DetrImageProcessor, DetrForObjectDetection

# Initialize Flask app and models
app = Flask(__name__)

# Load the YOLO model
yolo_model = torch.hub.load("ultralytics/yolov5", "yolov5s", force_reload=True)

# Load the Hugging Face model and processor for DETR
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
detr_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

# Configure Detectron2 model
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2  # Adjusted threshold for Detectron2
cfg.MODEL.DEVICE = "cpu"
detectron_predictor = DefaultPredictor(cfg)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Function to preprocess the image for better contrast
def preprocess_image(image_path):
    # Load the image using OpenCV
    image = cv2.imread(image_path)

    # Apply contrast adjustment using the specified curve
    lookup_table = np.arange(256, dtype=np.uint8)
    # Adjust the lower and upper bounds for the s-curve
    for i in range(256):
        if i < 40:
            lookup_table[i] = 0
        elif i > 215:
            lookup_table[i] = 255
        else:
            lookup_table[i] = int((i - 40) * (255 / (255 - 40)))

    # Apply the lookup table to adjust contrast
    contrast_image = cv2.LUT(image, lookup_table)

    # Save the modified image
    temp_image_path = image_path.replace("uploads/", "uploads/processed_")
    cv2.imwrite(temp_image_path, contrast_image)

    return temp_image_path

def count_with_yolov5(image_path, threshold=0.125):  # Set threshold as a parameter
    results = yolo_model(image_path)
    boxes = results.xyxy[0].cpu().numpy()
    return [{"box": [int(b[0]), int(b[1]), int(b[2]), int(b[3])], "label": f"{i+1}"} for i, b in enumerate(boxes) if b[4] > threshold]

def count_with_detectron2(image_path, threshold=0.3):  # Set threshold as a parameter
    image = cv2.imread(image_path)
    if image is None:
        return []
    outputs = detectron_predictor(image)
    instances = outputs["instances"].to("cpu")
    boxes = instances.pred_boxes if instances.has("pred_boxes") else []
    return [{"box": [int(b[0]), int(b[1]), int(b[2]), int(b[3])], "label": f"{i+1}"} for i, b in enumerate(boxes) if instances.scores[i] > threshold]

def count_with_huggingface(image_path, threshold=0.02):  # Set threshold as a parameter
    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = detr_model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]])  
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=threshold)[0]
    boxes = results["boxes"]
    return [{"box": [int(b[0]), int(b[1]), int(b[2]), int(b[3])], "label": f"DETR [{i+1}]"} for i, b in enumerate(boxes)]


# Updated drawing logic to draw unique counts for each detection box
def draw_default(image_path, yolov5_detections, detectron2_detections, huggingface_detections):
    # Decide which detections to draw based on counts
    all_detections = {
        "YOLOv5": yolov5_detections,
        "Detectron2": detectron2_detections,
        "Hugging Face": huggingface_detections
    }
    # Select the algorithm with the most detections
    best_algo = max(all_detections.items(), key=lambda x: len(x[1]))
    detections_to_draw = best_algo[1]

    annotated_image = Image.open(image_path)
    draw = ImageDraw.Draw(annotated_image)

    # Load a larger font
    font_size = 20  # Increase font size here (3x larger)
    font = ImageFont.truetype("arial.ttf", font_size)

    # Draw bounding boxes and unique labels (1 to n)
    for index, det in enumerate(detections_to_draw, start=1):  # Start numbering from 1
        box = det["box"]
        draw.rectangle(box, outline="green", width=2)

        # Prepare the label with the unique count
        label = str(index)  # Display the unique index

        # Calculate position for centered text using textbbox
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]  # Calculate width
        text_height = text_bbox[3] - text_bbox[1]  # Calculate height

        # Center the text
        text_x = box[0] + (box[2] - box[0]) / 2 - text_width / 2
        text_y = box[1] + (box[3] - box[1]) / 2 - text_height / 2

        # Calculate the shadow offset (e.g., 2 pixels)
        shadow_offset = -1

        # Draw the shadow text with a darker color and offset from the original text
        draw.text((text_x - shadow_offset, text_y - shadow_offset), label, fill="#000000", font=font)
        
        # Draw the original text with the desired color and font
        draw.text((text_x, text_y), label, fill="#55ff55", font=font)

    return annotated_image

def draw_single(image_path, detections):
    # Simplified version for individual threshold runs
    annotated_image = Image.open(image_path)
    draw = ImageDraw.Draw(annotated_image)
    font = ImageFont.truetype("arial.ttf", 20)

    for index, det in enumerate(detections, start=1):
        box = det["box"]
        draw.rectangle(box, outline="green", width=2)
        label = str(index)
        text_x = box[0] + (box[2] - box[0]) / 2 - 10
        text_y = box[1] + (box[3] - box[1]) / 2 - 10
        draw.text((text_x, text_y), label, fill="#55ff55", font=font)

    return annotated_image


@app.route("/", methods=["GET", "POST"])
def upload_image():
    if request.method == "POST":
        file = request.files.get("file")
        if not file:
            return jsonify({"error": "No file uploaded"}), 400

        try:
            img_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{uuid.uuid4()}.jpg")
            file.save(img_path)

            # Preprocess the image
            processed_img_path = preprocess_image(img_path)

            # Default detections
            yolov5_detections = count_with_yolov5(processed_img_path)
            detectron2_detections = count_with_detectron2(processed_img_path)
            huggingface_detections = count_with_huggingface(processed_img_path)

            # Draw default annotated image
            annotated_image = draw_default(img_path, yolov5_detections, detectron2_detections, huggingface_detections)
            annotated_path = os.path.join(app.config["UPLOAD_FOLDER"], f"annotated_{uuid.uuid4()}.jpg")
            annotated_image.save(annotated_path)

            # Run additional thresholds
            thresholds = [0.9, 0.7, 0.5, 0.3, 0.1]
            additional_images = []

            for threshold in thresholds:
                # Run each engine with the current threshold
                yolov5_results = count_with_yolov5(processed_img_path, threshold=threshold)
                detectron2_results = count_with_detectron2(processed_img_path, threshold=threshold)
                huggingface_results = count_with_huggingface(processed_img_path, threshold=threshold)

                # Generate result image for each threshold
                yolov5_img = draw_single(img_path, yolov5_results)
                detectron2_img = draw_single(img_path, detectron2_results)
                huggingface_img = draw_single(img_path, huggingface_results)

                # Save each result image and track it with engine type and threshold
                yolov5_img_path = os.path.join(app.config["UPLOAD_FOLDER"], f"yolo_{threshold}.jpg")
                detectron2_img_path = os.path.join(app.config["UPLOAD_FOLDER"], f"detectron_{threshold}.jpg")
                huggingface_img_path = os.path.join(app.config["UPLOAD_FOLDER"], f"huggingface_{threshold}.jpg")
                yolov5_img.save(yolov5_img_path)
                detectron2_img.save(detectron2_img_path)
                huggingface_img.save(huggingface_img_path)

                additional_images.extend([
                    {"engine": "YOLOv5", "threshold": threshold, "image_url": f"/uploads/{os.path.basename(yolov5_img_path)}"},
                    {"engine": "Detectron2", "threshold": threshold, "image_url": f"/uploads/{os.path.basename(detectron2_img_path)}"},
                    {"engine": "HuggingFace DETR", "threshold": threshold, "image_url": f"/uploads/{os.path.basename(huggingface_img_path)}"}
                ])

            # Generate HTML table
            html_table = generate_html_table(additional_images)

            # JSON response for frontend
            return jsonify({
                "yolov5_count": len(yolov5_detections),
                "detectron_count": len(detectron2_detections),
                "huggingface_count": len(huggingface_detections),
                "image_url": f"/uploads/{os.path.basename(annotated_path)}",
                "additional_images": html_table
            })

        except Exception as e:
            return jsonify({"error": f"Error saving or processing image: {str(e)}"}), 400

    return '''
    <!doctype html>
    <title>Upload Image for Detection</title>
    <h1>Upload an Image</h1>
    <form id="upload-form" method="post" enctype="multipart/form-data">
      <input type="file" name="file" required>
      <input type="submit" value="Upload and Detect">
    </form>
    <div id="result"></div>

    <script>
        document.getElementById("upload-form").onsubmit = function() {
            const formData = new FormData(this);
            fetch(this.action, {
                method: this.method,
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                let resultDiv = document.getElementById("result");
                resultDiv.innerHTML = `<img src="${data.image_url}" /><br />` + data.additional_images;
            })
            .catch(error => console.error('Error:', error));
            return false;
        }
    </script>
    '''

def generate_html_table(image_data):
    html = "<table><tr><th>Engine</th><th>Threshold</th><th>Result Image</th></tr>"
    for item in image_data:
        html += f"<tr><td>{item['engine']}</td><td>{item['threshold']}</td><td><img src='{item['image_url']}' width='200'></td></tr>"
    html += "</table>"
    return html

@app.route("/uploads/<path:filename>", methods=["GET"])
def send_image(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

if __name__ == "__main__":
    app.run(debug=True)