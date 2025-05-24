# app.py (TFLite)

import os
import json
import numpy as np
import cv2
import tensorflow as tf
# Note: We are no longer using keras.models.load_model
from flask import Flask, request, render_template, redirect, url_for
import base64
import matplotlib.pyplot as plt

# --- Initialize Flask App ---
app = Flask(__name__)

# --- Configuration ---
TFLITE_MODEL_PATH = 'model.tflite' # <-- Using the new model
CONFIG_PATH = 'model_config.json'

# --- Global Variables ---
interpreter = None
config = None
input_details = None
output_details = None

def load_model_and_config():
    """
    Loads the TFLite model and configuration.
    """
    global interpreter, config, input_details, output_details
    print("✅ Loading TFLite model and configuration...")
    
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)

    # Load the TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
    interpreter.allocate_tensors()

    # Get input and output tensor details.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print("✅ TFLite model and configuration loaded successfully.")

# Load the model once when the app starts
load_model_and_config()

def run_prediction(image_array):
    """
    Takes a NumPy image array, runs prediction with the TFLite interpreter,
    and returns an image with bounding boxes.
    """
    global interpreter, config, input_details, output_details
    
    # --- Prepare Image ---
    original_height, original_width = image_array.shape[:2]
    # TFLite model expects a specific input shape and type
    input_shape = input_details[0]['shape']
    IMG_HEIGHT, IMG_WIDTH = input_shape[1], input_shape[2]
    
    image_resized = cv2.resize(image_array, (IMG_WIDTH, IMG_HEIGHT))
    # Check the input type the model expects (usually float32)
    input_type = input_details[0]['dtype']
    image_normalized = np.array(image_resized, dtype=input_type) / 255.0
    image_batch = np.expand_dims(image_normalized, axis=0)

    # --- Run Prediction with TFLite Interpreter ---
    interpreter.set_tensor(input_details[0]['index'], image_batch)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])[0]

    # --- Decode Predictions & Apply NMS (same logic as before) ---
    detected_boxes_list = []
    for r in range(config['GRID_H']):
        for c in range(config['GRID_W']):
            cell_pred = predictions[r, c, :]
            pred_conf_logit = cell_pred[4]
            pred_confidence = tf.sigmoid(pred_conf_logit).numpy()
            if pred_confidence < 0.3:
                continue
            
            # ... (The rest of the decoding and NMS logic is identical to the previous version)
            pred_x_cell = tf.sigmoid(cell_pred[0]).numpy()
            pred_y_cell = tf.sigmoid(cell_pred[1]).numpy()
            pred_w_img = tf.sigmoid(cell_pred[2]).numpy()
            pred_h_img = tf.sigmoid(cell_pred[3]).numpy()
            pred_class_logits = cell_pred[5:]
            pred_class_probs = tf.nn.softmax(pred_class_logits).numpy()
            predicted_class_id_val = np.argmax(pred_class_probs)
            class_score = pred_class_probs[predicted_class_id_val] * pred_confidence

            if class_score < 0.3:
                continue

            center_x_img_norm = (pred_x_cell + c) / config['GRID_W']
            center_y_img_norm = (pred_y_cell + r) / config['GRID_H']
            xmin_norm = center_x_img_norm - (pred_w_img / 2.0)
            ymin_norm = center_y_img_norm - (pred_h_img / 2.0)
            xmax_norm = center_x_img_norm + (pred_w_img / 2.0)
            ymax_norm = center_y_img_norm + (pred_h_img / 2.0)
            xmin = int(xmin_norm * original_width)
            ymin = int(ymin_norm * original_height)
            xmax = int(xmax_norm * original_width)
            ymax = int(ymax_norm * original_height)
            detected_boxes_list.append([xmin, ymin, xmax, ymax, predicted_class_id_val, class_score])

    if not detected_boxes_list:
        return image_array

    # NMS logic
    boxes_for_nms = np.array([[b[0], b[1], b[2], b[3]] for b in detected_boxes_list], dtype=np.float32)
    scores_for_nms = np.array([b[5] for b in detected_boxes_list], dtype=np.float32)
    class_ids_for_nms = np.array([b[4] for b in detected_boxes_list])
    final_boxes_list = []
    unique_class_ids = np.unique(class_ids_for_nms)
    for class_id_val_iter in unique_class_ids:
        indices = np.where(class_ids_for_nms == class_id_val_iter)[0]
        class_boxes = tf.gather(boxes_for_nms, indices)
        class_scores = tf.gather(scores_for_nms, indices)
        selected_indices = tf.image.non_max_suppression(boxes=class_boxes, scores=class_scores, max_output_size=50, iou_threshold=0.4).numpy()
        for idx in selected_indices:
            final_boxes_list.append(detected_boxes_list[indices[idx]])
            
    # --- Draw Boxes on Image ---
    output_image_draw = image_array.copy()
    id_to_class = {int(k): v for k, v in config['id_to_class'].items()}
    for xmin, ymin, xmax, ymax, class_id, score in final_boxes_list:
        label = id_to_class.get(class_id, "Unknown")
        hue = class_id / float(config['NUM_CLASSES'])
        color_rgb_norm = plt.cm.hsv(hue)[:3]
        color_bgr_cv = tuple(int(c * 255) for c in reversed(color_rgb_norm))
        cv2.rectangle(output_image_draw, (xmin, ymin), (xmax, ymax), color_bgr_cv, 2)
        label_text = f"{label}: {score:.2f}"
        cv2.putText(output_image_draw, label_text, (xmin, ymin - 10 if ymin > 20 else ymin + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr_cv, 2)

    return output_image_draw


# --- Flask Routes (These remain the same) ---
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filestr = file.read()
        npimg = np.frombuffer(filestr, np.uint8)
        image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        result_image_array = run_prediction(image)
        _, buffer = cv2.imencode('.jpg', result_image_array)
        result_image_base64 = base64.b64encode(buffer).decode('utf-8')
        return render_template('result.html', result_image=result_image_base64)
    return redirect(url_for('home'))

# --- Main Execution (for local development ONLY) ---
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)