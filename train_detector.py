# train_detector.py

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2 # OpenCV for image loading, resizing, and drawing
import os
from sklearn.model_selection import train_test_split
import shutil # For cleaning up directories if needed

# --- Configuration ---
IMG_WIDTH = 224
IMG_HEIGHT = 224

# These will be determined by the model's backbone output after model definition
GRID_H = 0
GRID_W = 0

# These will be set after loading CSV and defining the model
NUM_CLASSES = 0
OUTPUT_FEATURES_PER_CELL = 0 # 4 (coords) + 1 (confidence) + NUM_CLASSES

# Loss function hyperparameters (inspired by YOLOv1)
LAMBDA_COORD = 5.0
LAMBDA_NOOBJ = 0.5

# Training params
BATCH_SIZE = 8 # Start with a smaller batch size for local CPU training; adjust based on your RAM
EPOCHS = 30    # Start with fewer epochs for local CPU; training will be slow

# --- File and Directory Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Gets the directory where the script is located
ANNOTATIONS_CSV_PATH = os.path.join(BASE_DIR, '_annotations.csv')
IMAGE_DIR_PATH = os.path.join(BASE_DIR, 'dataset_images')
MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'best_contact_lens_detector.keras') # Keras v3 native format

# --- Global Variables (will be populated later) ---
class_to_id = None
id_to_class = None


def load_and_preprocess_annotations():
    global NUM_CLASSES, class_to_id, id_to_class, image_annotations

    if not os.path.exists(ANNOTATIONS_CSV_PATH):
        print(f"Error: Annotations file not found at {ANNOTATIONS_CSV_PATH}")
        return False
    if not os.path.isdir(IMAGE_DIR_PATH):
        print(f"Error: Image directory not found at {IMAGE_DIR_PATH}")
        return False

    annotations_df = pd.read_csv(ANNOTATIONS_CSV_PATH)

    class_names_list = sorted(annotations_df['class'].unique())
    NUM_CLASSES = len(class_names_list)
    class_to_id = {name: i for i, name in enumerate(class_names_list)}
    id_to_class = {i: name for i, name in enumerate(class_names_list)}

    print(f"Found {NUM_CLASSES} classes: {class_names_list}")
    print("Class to ID mapping:", class_to_id)

    for col in ['xmin', 'ymin', 'xmax', 'ymax', 'width', 'height']:
        annotations_df[col] = pd.to_numeric(annotations_df[col], errors='coerce')
    annotations_df.dropna(subset=['xmin', 'ymin', 'xmax', 'ymax', 'width', 'height', 'class', 'filename'], inplace=True)

    annotations_df = annotations_df[annotations_df['xmin'] < annotations_df['xmax']]
    annotations_df = annotations_df[annotations_df['ymin'] < annotations_df['ymax']]
    annotations_df = annotations_df[annotations_df['width'] > 0]
    annotations_df = annotations_df[annotations_df['height'] > 0]
    
    image_annotations = {}
    for filename_csv, group in annotations_df.groupby('filename'):
        image_path = os.path.join(IMAGE_DIR_PATH, filename_csv)
        if not os.path.exists(image_path):
            print(f"Warning: Image file {filename_csv} listed in CSV but not found in {IMAGE_DIR_PATH}. Skipping.")
            continue

        img_width_csv = group['width'].iloc[0]
        img_height_csv = group['height'].iloc[0]

        boxes = []
        for _, row in group.iterrows():
            xmin = row['xmin'] / img_width_csv
            ymin = row['ymin'] / img_height_csv
            xmax = row['xmax'] / img_width_csv
            ymax = row['ymax'] / img_height_csv
            
            # Ensure normalized coordinates are valid
            if not (0 <= xmin < xmax <= 1 and 0 <= ymin < ymax <= 1):
                print(f"Warning: Invalid normalized coordinates for {filename_csv}, box {row['class']}. Skipping box.")
                continue

            boxes.append({
                'class_id': class_to_id[row['class']],
                'xmin': xmin,
                'ymin': ymin,
                'xmax': xmax,
                'ymax': ymax
            })
        if boxes:
             image_annotations[image_path] = {
                'boxes': boxes,
                'original_width': img_width_csv, # Store for reference if needed, though not used directly later
                'original_height': img_height_csv
            }
    
    print(f"Processed annotations for {len(image_annotations)} images with valid data.")
    if not image_annotations:
        print("Error: No valid annotations could be loaded. Please check your CSV and image files.")
        return False
    return True


def build_detection_model(input_shape, num_classes_model_param, grid_h_param, grid_w_param):
    global GRID_H, GRID_W, OUTPUT_FEATURES_PER_CELL # Allow modification of global vars

    input_tensor = keras.Input(shape=input_shape)

    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x) # 224x224 -> 112x112

    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x) # 112x112 -> 56x56

    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x) # 56x56 -> 28x28

    x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x) # 28x28 -> 14x14

    x = layers.Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    backbone_output = layers.MaxPooling2D((2, 2))(x) # 14x14 -> 7x7

    # Dynamically set GRID_H, GRID_W from backbone_output shape
    _, GRID_H_actual, GRID_W_actual, _ = backbone_output.shape
    GRID_H = int(GRID_H_actual) # Cast to int
    GRID_W = int(GRID_W_actual) # Cast to int
    print(f"Model backbone output feature map HxW: {GRID_H}x{GRID_W}")
    
    OUTPUT_FEATURES_PER_CELL = 4 + 1 + num_classes_model_param # x, y, w, h, conf, class_probs
    
    output_tensor = layers.Conv2D(OUTPUT_FEATURES_PER_CELL, (1, 1), activation='linear', name='detection_output')(backbone_output)
    
    model = keras.Model(inputs=input_tensor, outputs=output_tensor)
    return model


def preprocess_image_and_target(image_path, annotation_data):
    global GRID_H, GRID_W, OUTPUT_FEATURES_PER_CELL, NUM_CLASSES # Access global vars

    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Could not read image {image_path}. Skipping.")
        return None, None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
    image_normalized = image_resized / 255.0

    target_tensor = np.zeros((GRID_H, GRID_W, OUTPUT_FEATURES_PER_CELL), dtype=np.float32)

    for box_info in annotation_data['boxes']:
        xmin_norm, ymin_norm, xmax_norm, ymax_norm = box_info['xmin'], box_info['ymin'], box_info['xmax'], box_info['ymax']
        class_id = box_info['class_id']

        box_w_img = xmax_norm - xmin_norm
        box_h_img = ymax_norm - ymin_norm
        cx_img = xmin_norm + box_w_img / 2.0
        cy_img = ymin_norm + box_h_img / 2.0

        grid_x_idx = min(int(cx_img * GRID_W), GRID_W - 1)
        grid_y_idx = min(int(cy_img * GRID_H), GRID_H - 1)
        
        if target_tensor[grid_y_idx, grid_x_idx, 4] == 1: # If confidence is already 1, skip
            continue

        x_center_cell = (cx_img * GRID_W) - grid_x_idx
        y_center_cell = (cy_img * GRID_H) - grid_y_idx

        target_tensor[grid_y_idx, grid_x_idx, 0] = x_center_cell
        target_tensor[grid_y_idx, grid_x_idx, 1] = y_center_cell
        target_tensor[grid_y_idx, grid_x_idx, 2] = box_w_img
        target_tensor[grid_y_idx, grid_x_idx, 3] = box_h_img
        target_tensor[grid_y_idx, grid_x_idx, 4] = 1.0
        target_tensor[grid_y_idx, grid_x_idx, 5 + class_id] = 1.0
        
    return image_normalized, target_tensor


def create_tf_dataset(image_paths_list, annotations_dict, batch_size_param):
    processed_images_list = []
    processed_targets_list = []

    for img_path in image_paths_list:
        if img_path in annotations_dict:
            annots = annotations_dict[img_path]
            img_data, tgt_data = preprocess_image_and_target(img_path, annots)
            if img_data is not None and tgt_data is not None:
                processed_images_list.append(img_data)
                processed_targets_list.append(tgt_data)
    
    if not processed_images_list:
        return None

    X_data = np.array(processed_images_list)
    Y_data = np.array(processed_targets_list)
    
    dataset = tf.data.Dataset.from_tensor_slices((X_data, Y_data))
    dataset = dataset.shuffle(buffer_size=len(X_data)).batch(batch_size_param).prefetch(tf.data.AUTOTUNE)
    return dataset

def yolo_style_loss(y_true, y_pred):
    true_box_coords = y_true[..., 0:4]
    true_confidence = y_true[..., 4:5]
    true_class_probs = y_true[..., 5:]

    pred_box_coords_raw = y_pred[..., 0:4]
    pred_confidence_logits = y_pred[..., 4:5]
    pred_class_logits = y_pred[..., 5:]

    pred_box_xy = tf.sigmoid(pred_box_coords_raw[..., 0:2])
    pred_box_wh = tf.sigmoid(pred_box_coords_raw[..., 2:4])

    obj_mask = true_confidence
    noobj_mask = 1.0 - obj_mask

    xy_loss = tf.reduce_sum(tf.square(true_box_coords[..., 0:2] - pred_box_xy) * obj_mask, axis=[1,2,3])
    wh_loss = tf.reduce_sum(tf.square(tf.sqrt(true_box_coords[..., 2:4] + 1e-6) - tf.sqrt(pred_box_wh + 1e-6)) * obj_mask, axis=[1,2,3])
    coord_loss = LAMBDA_COORD * (xy_loss + wh_loss)

    obj_confidence_loss = tf.reduce_sum(
        tf.keras.losses.binary_crossentropy(true_confidence, pred_confidence_logits, from_logits=True) * obj_mask[...,0],
        axis=[1,2]
    )
    noobj_confidence_loss = LAMBDA_NOOBJ * tf.reduce_sum(
        tf.keras.losses.binary_crossentropy(tf.zeros_like(true_confidence), pred_confidence_logits, from_logits=True) * noobj_mask[...,0], # Target for noobj is 0
        axis=[1,2]
    )
    confidence_loss = obj_confidence_loss + noobj_confidence_loss

    class_loss = tf.reduce_sum(
        tf.keras.losses.categorical_crossentropy(true_class_probs, pred_class_logits, from_logits=True) * obj_mask[...,0],
        axis=[1,2]
    )

    total_loss = tf.reduce_mean(coord_loss + confidence_loss + class_loss)
    return total_loss


def predict_and_display(model_to_use, image_path_or_array, confidence_threshold=0.3, iou_threshold=0.4):
    global GRID_H, GRID_W, NUM_CLASSES, id_to_class # Access global configuration

    if isinstance(image_path_or_array, str):
        if not os.path.exists(image_path_or_array):
            print(f"Error: Image for prediction not found at {image_path_or_array}")
            return
        image = cv2.imread(image_path_or_array)
        if image is None:
            print(f"Error: Could not load image from {image_path_or_array}")
            return
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image_path_or_array
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    original_height, original_width = image.shape[:2]
    
    image_resized = cv2.resize(image_rgb, (IMG_WIDTH, IMG_HEIGHT))
    image_normalized = image_resized / 255.0
    image_batch = np.expand_dims(image_normalized, axis=0)

    predictions = model_to_use.predict(image_batch)[0]

    detected_boxes_list = []
    for r in range(GRID_H):
        for c in range(GRID_W):
            cell_pred = predictions[r, c, :]
            pred_x_cell_raw, pred_y_cell_raw, pred_w_img_raw, pred_h_img_raw = cell_pred[0:4]
            pred_conf_logit = cell_pred[4]
            pred_class_logits = cell_pred[5:]

            pred_x_cell = tf.sigmoid(pred_x_cell_raw).numpy()
            pred_y_cell = tf.sigmoid(pred_y_cell_raw).numpy()
            pred_w_img = tf.sigmoid(pred_w_img_raw).numpy()
            pred_h_img = tf.sigmoid(pred_h_img_raw).numpy()
            pred_confidence = tf.sigmoid(pred_conf_logit).numpy()
            
            if pred_confidence < confidence_threshold:
                continue

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
        print("No objects detected with confidence >", confidence_threshold)
        plt.imshow(image_rgb)
        plt.title("No Detections")
        plt.axis('off')
        plt.show()
        return

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

    output_image_draw = image.copy()
    for xmin, ymin, xmax, ymax, class_id_val, score_val in final_boxes_list:
        label = id_to_class[class_id_val]
        
        # Generate distinct colors for classes
        hue = class_id_val / float(NUM_CLASSES if NUM_CLASSES > 0 else 1)
        color_rgb_norm = plt.cm.hsv(hue)[:3] # Get RGB tuple (0-1 range)
        color_bgr_cv = tuple(int(c * 255) for c in reversed(color_rgb_norm)) # BGR for OpenCV, 0-255

        cv2.rectangle(output_image_draw, (xmin, ymin), (xmax, ymax), color_bgr_cv, 2)
        label_text = f"{label}: {score_val:.2f}"
        cv2.putText(output_image_draw, label_text, (xmin, ymin - 10 if ymin > 20 else ymin + 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr_cv, 2)

    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(output_image_draw, cv2.COLOR_BGR2RGB))
    plt.title("Detected Faults")
    plt.axis('off')
    plt.show()


# --- Main Execution ---
if __name__ == "__main__":
    print("--- Contact Lens Defect Detection ---")
    print(f"Using TensorFlow version: {tf.__version__}")
    print(f"Attempting to load annotations from: {ANNOTATIONS_CSV_PATH}")
    print(f"Attempting to load images from: {IMAGE_DIR_PATH}")

    if not load_and_preprocess_annotations():
        print("Failed to load annotations or dataset. Exiting.")
        exit()

    all_image_paths_main = list(image_annotations.keys())
    if not all_image_paths_main:
        print("No image paths found after processing annotations. Exiting.")
        exit()
        
    # Split data: 80% train, 20% validation
    train_paths, val_paths = train_test_split(all_image_paths_main, test_size=0.2, random_state=42)
    print(f"Total images: {len(all_image_paths_main)}, Training images: {len(train_paths)}, Validation images: {len(val_paths)}")

    # Build model - GRID_H, GRID_W, OUTPUT_FEATURES_PER_CELL will be set inside
    # Pass dummy grid_h, grid_w, they will be determined by the model architecture
    model_obj = build_detection_model(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), 
                                  num_classes_model_param=NUM_CLASSES, 
                                  grid_h_param=7, grid_w_param=7) # Initial guess for grid size
    model_obj.summary()
    print(f"Model configured with GRID_H={GRID_H}, GRID_W={GRID_W}, OUTPUT_FEATURES_PER_CELL={OUTPUT_FEATURES_PER_CELL}")

    # Create tf.data datasets
    print("Creating training dataset...")
    train_dataset_tf = create_tf_dataset(train_paths, image_annotations, BATCH_SIZE)
    print("Creating validation dataset...")
    val_dataset_tf = create_tf_dataset(val_paths, image_annotations, BATCH_SIZE)

    if train_dataset_tf is None:
        print("Failed to create training dataset. There might be no valid images/annotations. Exiting.")
        exit()

    model_obj.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4), loss=yolo_style_loss)

    print(f"\n--- Starting Model Training for {EPOCHS} epochs ---")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Model will be saved to: {MODEL_SAVE_PATH}")
    
    # Callback to save the best model
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=MODEL_SAVE_PATH,
        save_weights_only=False, # Save the entire model
        monitor='val_loss' if val_dataset_tf else 'loss',
        mode='min',
        save_best_only=True,
        verbose=1
    )

    try:
        history = model_obj.fit(
            train_dataset_tf,
            epochs=EPOCHS,
            validation_data=val_dataset_tf,
            callbacks=[model_checkpoint_callback]
        )
        print("\n--- Training Finished ---")

        # Plot training history
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Loss')
        if val_dataset_tf and 'val_loss' in history.history:
            plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.suptitle("Training Metrics")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make space for suptitle
        plt.show()

        # Load the best saved model for prediction example
        print(f"\nLoading best saved model from {MODEL_SAVE_PATH} for a prediction example...")
        if os.path.exists(MODEL_SAVE_PATH):
            trained_model = keras.models.load_model(MODEL_SAVE_PATH, custom_objects={'yolo_style_loss': yolo_style_loss})
            
            # Predict on a sample image from validation set if available, else from training set
            sample_image_path_pred = val_paths[0] if val_paths else (train_paths[0] if train_paths else None)
            if sample_image_path_pred:
                print(f"\n--- Predicting on a sample image: {sample_image_path_pred} ---")
                predict_and_display(trained_model, sample_image_path_pred)
            else:
                print("No sample image available for prediction example.")
        else:
            print(f"Model file not found at {MODEL_SAVE_PATH}. Skipping prediction example.")

        # --- Save model configuration for the prediction script ---
        print("Saving model configuration to model_config.json...")
        import json
        config_to_save = {
            'id_to_class': id_to_class,
            'GRID_H': GRID_H,
            'GRID_W': GRID_W,
            'IMG_WIDTH': IMG_WIDTH,
            'IMG_HEIGHT': IMG_HEIGHT,
            'NUM_CLASSES': NUM_CLASSES
        }
        with open('model_config.json', 'w') as f:
            json.dump(config_to_save, f, indent=4)
        print("Configuration saved successfully.")

    except Exception as e:
        print(f"\nAn error occurred during training or prediction: {e}")
        import traceback
        traceback.print_exc()

    print("\n--- Script Finished ---")