import os
import cv2
import albumentations as A
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "data/raw/train")
OUTPUT_DIR = os.path.join(BASE_DIR, "data/augmented/transformer/train")

os.makedirs(OUTPUT_DIR, exist_ok=True)

transform = A.Compose([
    A.HorizontalFlip(p=0.5), # left-right reversal (50% chance)
    A.RandomBrightnessContrast(p=0.2), # brightness and contrast adjustment (20% chance)
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5) # shift, scale, and rotation (50% chance)
])
    
for category in ["real", "fake"]:
    input_folder = os.path.join(INPUT_DIR, category)
    output_folder = os.path.join(OUTPUT_DIR, category)
    os.makedirs(output_folder, exist_ok=True)

    for img_file in os.listdir(input_folder):
        img_path = os.path.join(input_folder, img_file)
        img = cv2.imread(img_path)
        augmented_img = transform(image=img)
        augmented_img = augmented_img["image"]
        cv2.imwrite(os.path.join(output_folder, "aug_" + img_file), augmented_img)

print("Augmentation for Transformer training complete")