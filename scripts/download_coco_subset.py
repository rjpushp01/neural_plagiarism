import os
import json
import urllib.request
import zipfile
import requests
from pycocotools.coco import COCO
from tqdm import tqdm
from PIL import Image

def download_coco_subset(num_images=30, output_dir='./test_images/original'):
    os.makedirs(output_dir, exist_ok=True)
    
    ann_dir = './coco_annotations'
    ann_file = os.path.join(ann_dir, 'annotations', 'captions_train2017.json')
    zip_path = os.path.join(ann_dir, 'annotations_trainval2017.zip')
    
    if not os.path.exists(ann_file):
        print("Downloading COCO annotations...")
        os.makedirs(ann_dir, exist_ok=True)
        if not os.path.exists(zip_path):
            urllib.request.urlretrieve('http://images.cocodataset.org/annotations/annotations_trainval2017.zip', zip_path)
            print("Download complete.")
        print("Extracting annotations...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(ann_dir)
        print("Extraction complete.")
    
    print("Loading COCO annotations...")
    coco = COCO(ann_file)
    img_ids = coco.getImgIds()
    
    downloaded_count = 0
    print(f"Downloading {num_images} images from COCO...")
    metadata = {}
    
    for img_id in tqdm(img_ids):
        img_info = coco.loadImgs(img_id)[0]
        image_url = img_info['coco_url']
        filename = f"coco_{img_id:012d}.jpg"
        filepath = os.path.join(output_dir, filename)
        
        try:
            image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
            # Resize image to 512x512 to match Stable Diffusion requirements
            image = image.resize((512, 512))
            image.save(filepath)
            
            ann_ids = coco.getAnnIds(imgIds=img_id)
            anns = coco.loadAnns(ann_ids)
            captions = [ann['caption'] for ann in anns]
            
            metadata[filename] = {
                'id': img_id,
                'url': image_url,
                'captions': captions
            }
            
            downloaded_count += 1
            if downloaded_count >= num_images:
                break
        except Exception as e:
            print(f"Failed to download {image_url}: {e}")
            continue
            
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=4)
        
    print(f"Downloaded {downloaded_count} images successfully to {output_dir}")

if __name__ == '__main__':
    download_coco_subset(30, './test_images/original')
