import os
import glob
from PIL import Image, ImageDraw, ImageFont
import sys

# Add parent directory to path to import watermarker
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from watermarker import InvisibleWatermarker

def apply_visible_watermark(image_path, output_path, text="COPYRIGHT"):
    img = Image.open(image_path).convert("RGBA")
    txt = Image.new("RGBA", img.size, (255, 255, 255, 0))
    d = ImageDraw.Draw(txt)
    
    # Try to load a generic font, or fallback to default
    try:
        font = ImageFont.truetype("arial.ttf", 60)
    except IOError:
        font = ImageFont.load_default()
        
    # Positioning text in the center
    text_bbox = d.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    width, height = img.size
    position = ((width - text_width) // 2, (height - text_height) // 2)
    
    # Draw text with some transparency
    d.text(position, text, fill=(255, 255, 255, 128), font=font)
    
    watermarked = Image.alpha_composite(img, txt)
    watermarked.convert("RGB").save(output_path)

def apply_watermarks(input_dir='./test_images/original'):
    visible_dir = './test_images/visible'
    invisible_dir = './test_images/invisible'
    
    os.makedirs(visible_dir, exist_ok=True)
    os.makedirs(invisible_dir, exist_ok=True)
    
    img_paths = glob.glob(os.path.join(input_dir, '*.jpg'))
    
    if not img_paths:
        print(f"No images found in {input_dir}")
        return

    print(f"Applying watermarks to {len(img_paths)} images...")
    
    # Initialize invisible watermarker from the main codebase
    invisible_watermarker = InvisibleWatermarker(wm_text='test', method='dwtDctSvd')
    
    for img_path in img_paths:
        filename = os.path.basename(img_path)
        
        # 1. Apply visible watermark
        vis_out = os.path.join(visible_dir, filename)
        apply_visible_watermark(img_path, vis_out, text="COPYRIGHT")
        
        # 2. Apply invisible watermark
        invis_out = os.path.join(invisible_dir, filename)
        invisible_watermarker.encode(img_path, invis_out)
        
    print("Watermarking complete.")

if __name__ == '__main__':
    apply_watermarks('./test_images/original')
