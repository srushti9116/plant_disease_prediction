import os
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

MODEL_PATH = "model/plant_disease_model.h5"
TEST_DIR = "dataset/test"
IMG_SIZE = (224, 224)   # use the same size you used when training
BATCH_SIZE = 32

print("🔹 TensorFlow version:", tf.__version__)
print("🔹 Looking for model at:", MODEL_PATH)

# -----------------------------
# 1. Load Model (NO COMPILE NEEDED)
# -----------------------------
try:
    print("\n⏳ Loading model ...")
    model = load_model(MODEL_PATH, compile=False)
    print("✅ Model loaded successfully.\n")
except Exception as e:
    print("❌ Failed to load model:")
    print(e)
    raise

# -----------------------------
# 2. Prepare Test Data
# -----------------------------
if not os.path.isdir(TEST_DIR):
    print(f"❌ Test directory not found: {TEST_DIR}")
    raise SystemExit

print("🔹 Using test directory:", TEST_DIR)

test_datagen = ImageDataGenerator(rescale=1.0/255)

test_generator = test_datagen.flow_from_directory(
    directory=TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',   # labels as one-hot vectors
    shuffle=False
)

num_samples = test_generator.samples
num_classes = test_generator.num_classes
print(f"\n🔹 Test samples: {num_samples}, Classes: {num_classes}")

if num_samples == 0:
    print("❌ No images found in dataset/test/. Cannot compute accuracy.")
    raise SystemExit

steps = math.ceil(num_samples / BATCH_SIZE)

# -----------------------------
# 3. Predict & Compute Accuracy
# -----------------------------
print("\n⏳ Running predictions on test set ...")
pred_probs = model.predict(test_generator, steps=steps)

pred_labels = np.argmax(pred_probs, axis=1)
true_labels = test_generator.classes[:len(pred_labels)]

# Overall accuracy
accuracy = (pred_labels == true_labels).mean()

print("\n======================================")
print("✔ Overall Test Accuracy: {:.2f}%".format(accuracy * 100))
print("======================================\n")

# -----------------------------
# 4. Per-Class Metrics
# -----------------------------
class_names = list(test_generator.class_indices.keys())

print("✔ Classification Report:\n")
print(classification_report(true_labels, pred_labels, target_names=class_names))

print("\n✔ Confusion Matrix:\n")
print(confusion_matrix(true_labels, pred_labels))
