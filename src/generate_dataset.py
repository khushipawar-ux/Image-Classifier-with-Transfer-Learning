import os
import math
import random
from PIL import Image, ImageDraw

def generate_shape_image(shape_type, size=224):
    """
    Generates a synthetic image of a geometric shape (circle, square, triangle)
    with randomized background, foreground colors, position, size, and rotation.
    """
    # 1. Generate random high-contrast background and foreground colors
    # We want to ensure there is enough contrast for the shape to be visible
    bg_color = (random.randint(200, 255), random.randint(200, 255), random.randint(200, 255))
    
    # Foreground color should be darker to provide good contrast
    fg_color = (random.randint(0, 100), random.randint(0, 100), random.randint(0, 100))
    
    # Create blank canvas
    img = Image.new("RGB", (size, size), bg_color)
    draw = ImageDraw.Draw(img)
    
    # 2. Add some random background lines (noise/clutter) to make it a bit realistic
    num_bg_lines = random.randint(2, 6)
    for _ in range(num_bg_lines):
        x0 = random.randint(0, size)
        y0 = random.randint(0, size)
        x1 = random.randint(0, size)
        y1 = random.randint(0, size)
        line_color = (random.randint(150, 220), random.randint(150, 220), random.randint(150, 220))
        draw.line([x0, y0, x1, y1], fill=line_color, width=random.randint(1, 3))
        
    # 3. Define shape properties
    center_x = size // 2 + random.randint(-20, 20)
    center_y = size // 2 + random.randint(-20, 20)
    shape_size = random.randint(40, 80) # radius/half-width
    
    if shape_type == "circle":
        # Draw a circle
        x0 = center_x - shape_size
        y0 = center_y - shape_size
        x1 = center_x + shape_size
        y1 = center_y + shape_size
        draw.ellipse([x0, y0, x1, y1], fill=fg_color, outline=None)
        
    elif shape_type == "square":
        # Draw a rotated square using trigonometry
        angle = random.uniform(0, 2 * math.pi)
        points = []
        for i in range(4):
            # A square has 4 corners at 90-degree intervals (pi/2 radians)
            theta = angle + i * (math.pi / 2)
            # Distance from center to corners is shape_size * sqrt(2) to keep side length proportional
            px = center_x + int(shape_size * math.sqrt(2) * math.cos(theta))
            py = center_y + int(shape_size * math.sqrt(2) * math.sin(theta))
            points.append((px, py))
        draw.polygon(points, fill=fg_color, outline=None)
        
    elif shape_type == "triangle":
        # Draw a rotated equilateral triangle using trigonometry
        angle = random.uniform(0, 2 * math.pi)
        points = []
        for i in range(3):
            # An equilateral triangle has 3 corners at 120-degree intervals (2*pi/3 radians)
            theta = angle + i * (2 * math.pi / 3)
            px = center_x + int(shape_size * 1.2 * math.cos(theta))
            py = center_y + int(shape_size * 1.2 * math.sin(theta))
            points.append((px, py))
        draw.polygon(points, fill=fg_color, outline=None)
        
    return img

def build_dataset(base_dir="data", train_samples=500, val_samples=100):
    """
    Builds the dataset directories and generates images for train and val splits.
    """
    shapes = ["circle", "square", "triangle"]
    splits = {
        "train": train_samples,
        "val": val_samples
    }
    
    print("Generating synthetic shapes dataset...")
    for split, count in splits.items():
        print(f"Generating {count} images per class for split: {split}")
        for shape in shapes:
            class_dir = os.path.join(base_dir, split, shape)
            os.makedirs(class_dir, exist_ok=True)
            
            for i in range(count):
                img = generate_shape_image(shape)
                img_path = os.path.join(class_dir, f"{shape}_{i:04d}.jpg")
                img.save(img_path, "JPEG")
                
    print("Dataset generation complete! Images saved under target folder:", base_dir)

if __name__ == "__main__":
    # Seed for reproducibility
    random.seed(42)
    build_dataset(base_dir="data", train_samples=500, val_samples=100)
