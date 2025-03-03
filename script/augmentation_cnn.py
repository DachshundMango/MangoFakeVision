import os
import torchvision.transforms as transforms
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "data/raw/train")
OUTPUT_DIR = os.path.join(BASE_DIR, "data/augmented/cnn/train")

os.makedirs(OUTPUT_DIR, exist_ok=True)

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5), # left-right reversal
    transforms.RandomRotation(10), # 10 degree rotation
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1), # color jitter
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.8, 1.2)) # random crop and resize
])

for category in ["real", "fake"]:
    input_folder = os.path.join(INPUT_DIR, category)
    output_folder = os.path.join(OUTPUT_DIR, category)
    os.makedirs(output_folder, exist_ok=True)

    for img_file in os.listdir(input_folder):
        img_path = os.path.join(input_folder, img_file)
        img = Image.open(img_path)
        augmented_img = transform(img)
        augmented_img.save(os.path.join(output_folder, "aug_" + img_file))

print("Augmentation for CNN training complete")

