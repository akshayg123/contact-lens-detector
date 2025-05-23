# predict_image.py

import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import json
import argparse # To easily pass image paths from the command line

# --- Configuration ---
MODEL_PATH = 'best_contact_lens_detector.keras'
CONFIG_PATH = 'model_config.json'

# This loss function definition is needed by Keras to load the model,
# as it was a custom function used during training.
def yolo_style_loss(y_true, y_pred):
    # This is a placeholder for loading purposes. The actual logic is not
    # needed for inference, but the name must be registered.
    # The actual implementation from the training script can be copied here if needed,
    # but for pure prediction, it's not executed.
    return tf.constant(0.0)


# This is the full prediction and visualization function, adapted from train_detector.py
def predict_and_display(model_to_use, image_path, config, confidence_threshold=0.3, iou_threshold=0.4):
    """
    Loads an image, runs prediction, and displays the results with bounding boxes.
    """
    # Load configuration from the passed dictionary
    IMG_WIDTH = config['IMG_WIDTH']
    IMG_HEIGHT = config['IMG_HEIGHT']
    GRID_H = config['GRID_H']
    GRID_W = config['GRID_W']
    NUM_CLASSES = config['NUM_CLASSES']
    # JSON saves integer keys as strings, so we convert them back
    id_to_class = {int(k): v for k, v in config['id_to_class'].items()}

    # --- Image Loading and Preprocessing ---
    if not os.path.exists(image_path):
        print(f"❌ Error: Image for prediction not found at {image_path}")
        return
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Error: Could not load image from {image_path}")
        return
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_height, original_width = image.shape[:2]
    
    image_resized = cv2.resize(image_rgb, (IMG_WIDTH, IMG_HEIGHT))
    image_normalized = image_resized / 255.0
    image_batch = np.expand_dims(image_normalized, axis=0)

    # --- Prediction ---
    predictions = model_to_use.predict(image_batch)[0]

    # --- Decoding Predictions ---
    detected_boxes_list = []
    for r in range(GRID_H):
        for c in range(GRID_W):
            cell_pred = predictions[r, c, :]
            pred_x_cell_raw, pred_y_cell_raw, pred_w_img_raw, pred_h_img_raw = cell_pred[0:4]
            pred_conf_logit = cell_pred[4]
            pred_class_logits = cell_pred[5:]

            pred_confidence = tf.sigmoid(pred_conf_logit).numpy()
            
            if pred_confidence < confidence_threshold:
                continue

            pred_x_cell = tf.sigmoid(pred_x_cell_raw).numpy()
            pred_y_cell = tf.sigmoid(pred_y_cell_raw).numpy()
            pred_w_img = tf.sigmoid(pred_w_img_raw).numpy()
            pred_h_img = tf.sigmoid(pred_h_img_raw).numpy()
            pred_class_probs = tf.nn.softmax(pred_class_logits).numpy()
            predicted_class_id_val = np.argmax(pred_class_probs)
            class_score = pred_class_probs[predicted_class_id_val] * pred_confidence

            if class_score < confidence_threshold:
                continue

            center_x_img_norm = (pred_x_cell + c) / GRID_W
            center_y_img_norm = (pred_y_cell + r) / GRID_H
            
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
        print("ℹ️ No objects detected with confidence >", confidence_threshold)
        plt.imshow(image_rgb)
        plt.title("No Detections")
        plt.axis('off')
        plt.show()
        return

    # --- Non-Max Suppression (to remove overlapping boxes) ---
    boxes_for_nms = np.array([[b[0], b[1], b[2], b[3]] for b in detected_boxes_list], dtype=np.float32)
    scores_for_nms = np.array([b[5] for b in detected_boxes_list], dtype=np.float32)
    class_ids_for_nms = np.array([b[4] for b in detected_boxes_list])
    
    final_boxes_list = []
    unique_class_ids = np.unique(class_ids_for_nms)

    for class_id_val_iter in unique_class_ids:
        indices_for_class = np.where(class_ids_for_nms == class_id_val_iter)[0]
        class_boxes = tf.gather(boxes_for_nms, indices_for_class)
        class_scores = tf.gather(scores_for_nms, indices_for_class)
        
        selected_indices = tf.image.non_max_suppression(
            boxes=class_boxes, scores=class_scores,
            max_output_size=50, iou_threshold=iou_threshold
        ).numpy()
        
        for idx in selected_indices:
            original_idx = indices_for_class[idx]
            final_boxes_list.append(detected_boxes_list[original_idx])

    # --- Drawing Final Boxes and Displaying the Image ---
    output_image_draw = image.copy()
    for xmin, ymin, xmax, ymax, class_id_val, score_val in final_boxes_list:
        label = id_to_class[class_id_val]
        
        hue = class_id_val / float(NUM_CLASSES if NUM_CLASSES > 0 else 1)
        color_rgb_norm = plt.cm.hsv(hue)[:3]
        color_bgr_cv = tuple(int(c * 255) for c in reversed(color_rgb_norm))

        cv2.rectangle(output_image_draw, (xmin, ymin), (xmax, ymax), color_bgr_cv, 2)
        label_text = f"{label}: {score_val:.2f}"
        cv2.putText(output_image_draw, label_text, (xmin, ymin - 10 if ymin > 20 else ymin + 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr_cv, 2)

    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(output_image_draw, cv2.COLOR_BGR2RGB))
    plt.title("Detected Faults")
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    # Setup command-line argument parser
    parser = argparse.ArgumentParser(description="Detect faults in contact lens images.")
    parser.add_argument("image_path", type=str, help="Path to the image file for prediction.")
    args = parser.parse_args()

    # --- Load Configuration and Model ---
    if not os.path.exists(CONFIG_PATH):
        print(f"❌ Error: Configuration file not found at {CONFIG_PATH}. Please run the training script first.")
        exit()
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model file not found at {MODEL_PATH}. Please run the training script first.")
        exit()
        
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)

    print("✅ Loading model...")
    # We must pass the custom loss function to Keras so it knows what "yolo_style_loss" is
    trained_model = keras.models.load_model(MODEL_PATH, custom_objects={'yolo_style_loss': yolo_style_loss})
    print("✅ Model loaded successfully.")

    # --- Run Prediction ---
    predict_and_display(trained_model, args.image_path, config)