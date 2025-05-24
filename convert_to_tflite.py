# convert_to_tflite.py

import tensorflow as tf
import os

KERAS_MODEL_PATH = 'best_contact_lens_detector.keras'
TFLITE_MODEL_PATH = 'model.tflite'

if not os.path.exists(KERAS_MODEL_PATH):
    print(f"Error: Keras model not found at {KERAS_MODEL_PATH}")
else:
    print(f"Loading Keras model from {KERAS_MODEL_PATH}...")
    # We don't need the custom loss function for conversion
    model = tf.keras.models.load_model(KERAS_MODEL_PATH, compile=False)

    # Convert the model to TensorFlow Lite format
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # This enables quantization, which makes the model even smaller and faster
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()

    # Save the converted model to a file
    with open(TFLITE_MODEL_PATH, 'wb') as f:
        f.write(tflite_model)

    print("-" * 50)
    print(f"✅ Model successfully converted and saved to {TFLITE_MODEL_PATH}")
    print(f"Original Keras model size: {os.path.getsize(KERAS_MODEL_PATH) / 1024:.2f} KB")
    print(f"New TFLite model size: {os.path.getsize(TFLITE_MODEL_PATH) / 1024:.2f} KB")
    print("-" * 50)