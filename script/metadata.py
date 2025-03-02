import os
import pandas as pd
from pathlib import Path
from PIL import Image
from PIL.ExifTags import TAGS
import piexif

def get_exif_data(image_path):
	try:
		
		if not Path(image_path).exists():
			print(f"{image_path} does not exist.")
			return "unknown", "unknown"
		
		with Image.open(image_path) as img:
			
			exif_data = img._getexif()
			
			if exif_data is None:
				exif_bytes = img.info.get("exif")
				if exif_bytes:
					try:
						exif_data = piexif.load(exif_bytes).get("0th", {})
					except Exception:
						print(f"Failed to load EXIF via exif bytes {image_path}")
						exif_data = {}
				else:
					exif_data = {}

			if not exif_data:
				return "unknown", "unknown"
			
			exif = {TAGS.get(k, str(k)): v for k, v in exif_data.items() if isinstance(k, int)}

			capture_device = exif.get("Model", "unknown")
			compression = exif.get("Compression", "unknown")

			return capture_device, compression
	
	except Exception as e:
		print(f"exif data error {image_path}: {e}")
		return "unknown", "unknown"

def get_split_and_ground_truth(image_path):
	filepath = str(Path(image_path))
	if "train" in filepath.lower():
		split = "train"
	elif "valid" in filepath.lower():
		split = "valid"
	elif "test" in filepath.lower():
		split = "test"
	else:
		split = "unknown"
	
	if "real" in filepath.lower():
		ground_truth = 1
	elif "fake" in filepath.lower():
		ground_truth = 0
	else:
		ground_truth = -1
    
	return split, ground_truth

def is_valid_image(image_path):
	valid_extensions = [".png", ".gpg", ".jpeg", ".jpg"]
	return Path(image_path).suffix.lower() in valid_extensions

def extract_image_metadata(image_path):
	try:
		with Image.open(image_path) as img:
			filename = os.path.basename(image_path)
			filepath = os.path.dirname(image_path)
			format = img.format
			resolution = f"{img.width}x{img.height}"
			colour_mode = img.mode
			# 'source' is not applicable.
			channels = len(img.getbands())
			capture_device, compression = get_exif_data(image_path)
			file_size = os.path.getsize(image_path)
			split, ground_truth = get_split_and_ground_truth(image_path)
			ground_truth_metadata = "1 is real, 0 is fake, -1 is unknown"
			original_ground_truth = img.info.get("label", "unknown")

		return {
                "filename": filename,
                "filepath": filepath,
                "format": format,
                "resolution": resolution,
                "colour_mode": colour_mode,
                "channels": channels,
                "capture_device": capture_device,
                "compression": compression,
                "file_size": file_size,
                "split": split,
                "ground_truth": ground_truth,
                "ground_truth_metadata": ground_truth_metadata,
                "original_ground_truth": original_ground_truth
            }

	except Exception as e:
		print(f"Error processing {image_path}: {e}")
		return None

DATASET_PATHS = {
    "train": "data/raw/train",
    "valid": "data/raw/valid",
    "test": "data/raw/test"
}

OUTPUT_DIR = "data/metadata"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SAVE_INTERVAL = 100000
batch_count = 0

for split, dataset_path in DATASET_PATHS.items():
	
	metadata_list = []
	error_list = []

	for real_or_fake in ["real", "fake"]:
	
		target_folder = Path(dataset_path) / real_or_fake

		image_paths = [f for f in target_folder.rglob("*.*") if is_valid_image(f)]

		for image_path in image_paths:
			metadata = extract_image_metadata(image_path)
			if metadata:
				metadata_list.append(metadata)
				batch_count += 1
				
				if batch_count % SAVE_INTERVAL == 0:
					print(f"extracted {batch_count} metadata of images...")

			else:
				error_metadata = {
					"filename": os.path.basename(image_path),
					"filepath": os.path.dirname(image_path)
				}
				error_list.append(error_metadata)
	
	csv_path = Path(OUTPUT_DIR) / f"{split}_metadata.csv"
	error_path = Path(OUTPUT_DIR) / f"{split}_metadata_error.csv"

	if metadata_list:
		metadata_df = pd.DataFrame(metadata_list)
		metadata_df.to_csv(csv_path, index=False)
		print(f"Metadata saved to {csv_path}")
	
	if error_list:
		error_df = pd.DataFrame(error_list)
		error_df.to_csv(error_path, index=False)
		print(f"Error data saved to {error_path}")