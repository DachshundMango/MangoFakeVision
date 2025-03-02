import os
from pathlib import Path
import pandas as pd
from PIL import Image, ExifTags
import piexif

IMAGE_DIR = "./image"

metadata = []

def exif_data_for_img(file_path):
    try:

        file_path = Path(file_path)

        if not file_path.exists():
            print(f"File {file_path} does not exist")
            return "unknown", "unknown"
        
        with Image.open(file_path) as img:
            exif_data = img._getexif()
            
            if exif_data is None:
                exif_bytes = img.info.get("exif")
                if exif_bytes:
                    try:
                        exif_data = piexif.load(exif_bytes).get("0th",{})
                    except Exception:
                        print(f"Faild to load EXIF for {file_path}")
                        exif_data = {}
                else:
                    exif_data = {}

            if not exif_data:
                return "unknown", "unknown"

            exif = {ExifTags.TAGS.get(k, str(k)): v for k, v in exif_data.items() if isinstance(k, int)}   
            
            capture_device = exif.get("Model", "unknown")
            compression = exif.get("Compression", "unknown")
            
            return capture_device, compression
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return "unknown", "unknown"

def split_img(image_path):
    filepath = os.path.dirname(image_path) 
    if "train" in filepath:
        split = "train"
    elif "test" in filepath:
        split = "test" 
    elif "val" in filepath:
        split = "val" 
    else:
        split = "unknown"
    return split

def ground_truth_img(image_path):
    filepath_str =str(Path(image_path))
    if "real" in filepath_str.lower():
        ground_truth = 1
    elif "fake" in filepath_str.lower():
        ground_truth = 0
    else:
        ground_truth = -1
    return ground_truth

def is_valid_image(image_path):
    valid_extensions = [".png", ".jpg", ".jpeg"]
    return Path(image_path).suffix.lower() in valid_extensions

def get_image_metadata(image_path):

    try:
        with Image.open(image_path) as img:
            filename = os.path.basename(image_path) 
            filepath = os.path.dirname(image_path) 
            category = Path(filepath).parts[2] if len(Path(filepath).parts) > 2 else "unknown"
            format = img.format 
            resolution = f"{img.width}x{img.height}" 
            source = img.info.get("source", "")
            colour_mode = img.mode 
            channels = len(img.getbands()) 
            capture_device, compression = exif_data_for_img(image_path)
            file_size = os.path.getsize(image_path) 
            split = split_img(image_path)
            ground_truth = ground_truth_img(image_path)
            ground_truth_metadata = "1 is real, 0 is fake, -1 is unknown"
            original_ground_truth = img.info.get("label", "unknown")

            return {
                "filename": filename,
                "filepath": filepath,
                "category": category,
                "format": format,
                "resolution": resolution,
                "source": source,
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
        #print(f"Error processing {image_path}: {e}")
        return None


directory_path = Path(IMAGE_DIR)

image_files = [f for f in directory_path.rglob("*.*") if is_valid_image(f)]
metadata_list = []
error_list = []

SAVE_INTERVAL = 100000
batch_count = 0

for image_path in image_files:
    metadata = get_image_metadata(image_path)
    if metadata:
        metadata_list.append(metadata)
        batch_count += 1

        if batch_count % SAVE_INTERVAL == 0:
            print(f"extracted {batch_count} metadata of images...")
            metadata_df = pd.DataFrame(metadata_list)
    else:
        error_metadata = {
            "filename": os.path.basename(image_path),
            "filepath": os.path.dirname(image_path),
        }
        error_list.append(error_metadata)


if metadata_list:
    metadata_df = pd.DataFrame(metadata_list)
    metadata_df.to_csv(f"./image/local_img_metadata_total_{batch_count}.csv", index=False) 
    print(f"Metadata saved to local_img_metadata.csv, total: {batch_count}")

if error_metadata:
    error_df = pd.DataFrame(error_list)
    error_df.to_csv("./image/error_img_metadata.csv", index=False) 
    print(f"Error metadata saved to error_img_metadata.csv")


