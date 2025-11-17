import os
import shutil
import random

# ----------------------------- #
#   YOUR ACTUAL DATASET PATH    #
# ----------------------------- #

# This is your PlantVillage dataset folder
RAW_DATASET = r"E:\Downloads\archive\PlantVillage"


# This folder will be created automatically
OUTPUT_DATASET = r"E:\plant_disease_prediction\dataset"


# ----------------------------- #
#         SPLIT RATIOS          #
# ----------------------------- #

TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

assert abs((TRAIN_SPLIT + VAL_SPLIT + TEST_SPLIT) - 1.0) < 1e-6, "Splits must sum to 1!"

# ----------------------------- #
#     CREATE OUTPUT FOLDERS     #
# ----------------------------- #

for folder in ["train", "val", "test"]:
    os.makedirs(os.path.join(OUTPUT_DATASET, folder), exist_ok=True)

# ----------------------------- #
#         START PROCESSING      #
# ----------------------------- #

classes = os.listdir(RAW_DATASET)
classes = [c for c in classes if os.path.isdir(os.path.join(RAW_DATASET, c))]

print("Found classes:", classes)

for cls in classes:
    src_dir = os.path.join(RAW_DATASET, cls)
    
    # Get all image files
    images = [f for f in os.listdir(src_dir)
              if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    if len(images) < 3:
        print(f"⚠️ Skipping {cls} — not enough images.")
        continue

    random.shuffle(images)

    total = len(images)
    n_train = int(total * TRAIN_SPLIT)
    n_val   = int(total * VAL_SPLIT)
    n_test  = total - n_train - n_val

    train_images = images[:n_train]
    val_images   = images[n_train:n_train + n_val]
    test_images  = images[n_train + n_val:]

    print(f"\n📁 Class: {cls}")
    print(f"Total: {total}")
    print(f"Train: {len(train_images)} | Val: {len(val_images)} | Test: {len(test_images)}")

    # Create class folders
    train_dst = os.path.join(OUTPUT_DATASET, "train", cls)
    val_dst   = os.path.join(OUTPUT_DATASET, "val", cls)
    test_dst  = os.path.join(OUTPUT_DATASET, "test", cls)

    os.makedirs(train_dst, exist_ok=True)
    os.makedirs(val_dst, exist_ok=True)
    os.makedirs(test_dst, exist_ok=True)

    # Copy files
    for img in train_images:
        shutil.copy(os.path.join(src_dir, img), os.path.join(train_dst, img))

    for img in val_images:
        shutil.copy(os.path.join(src_dir, img), os.path.join(val_dst, img))

    for img in test_images:
        shutil.copy(os.path.join(src_dir, img), os.path.join(test_dst, img))

print("\n🎉 DONE! Dataset successfully split into:")
print("➡ TRAIN:", os.path.join(OUTPUT_DATASET, "train"))
print("➡ VAL:", os.path.join(OUTPUT_DATASET, "val"))
print("➡ TEST:", os.path.join(OUTPUT_DATASET, "test"))
