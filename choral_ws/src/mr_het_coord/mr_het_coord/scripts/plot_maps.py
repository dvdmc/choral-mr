import numpy as np
from PIL import Image, ImageDraw
import math
from scipy.ndimage import distance_transform_edt

def get_star_points(cx, cy, outer_radius, inner_radius=None):
    if inner_radius is None:
        inner_radius = outer_radius / 2.5
    points = []
    for i in range(10):
        angle = i * (360 / 10) - 90
        r = outer_radius if i % 2 == 0 else inner_radius
        x = cx + r * math.cos(math.radians(angle))
        y = cy + r * math.sin(math.radians(angle))
        points.append((x, y))
    return points

def save_mission_map(data, name, scale, radius):
    """Figure 1: Obstacles (Black) + Low Trav (Ochre) + Star (Depot) + Pink Circles (Tasks)"""
    h, w = data.shape
    # Start with white background
    mission_bg = np.ones((h, w, 3), dtype=np.uint8) * 255
    
    # Map layers: Obstacles and Low Traversability
    mission_bg[data == 0] = [0, 0, 0]                # Obstacles: Black
    mission_bg[data == 125] = [80, 70, 30]        # Low Trav: Brown

    # Scale and convert to PIL
    img = Image.fromarray(mission_bg).resize((w * scale, h * scale), Image.NEAREST).convert("RGB")
    draw = ImageDraw.Draw(img)
    
    added_depot = False
    for y in range(h):
        for x in range(w):
            if data[y, x] == 200:
                cx, cy = x * scale, y * scale
                if not added_depot:
                    # Green Depot Star
                    points = get_star_points(cx, cy, radius * 2)
                    draw.polygon(points, fill="#00A334", outline="#00A334")
                    added_depot = True
                else:
                    # Pink Task Circles
                    draw.ellipse((cx-radius, cy-radius, cx+radius, cy+radius), fill="#FE6AF0")
    img.save(f"{name}_mission.png")

def save_traversability_map(data, name, scale):
    """Figure 2: Pure environment map - Obstacles (Black), Low Trav (Ochre), Free (White)"""
    h, w = data.shape
    color_map = np.ones((h, w, 3), dtype=np.uint8) * 255
    color_map[data == 0] = [0, 0, 0]
    color_map[data == 125] = [183, 135, 65]
    
    img = Image.fromarray(color_map).resize((w * scale, h * scale), Image.NEAREST)
    img.save(f"{name}_traversability.png")

def save_sdf_map(data, name, scale):
    """Figure 3: SDF Gradient from obstacles (0)"""
    mask = data != 0
    sdf = distance_transform_edt(mask)
    if sdf.max() > 0:
        sdf_viz = (sdf / sdf.max() * 255).astype(np.uint8)
    else:
        sdf_viz = sdf.astype(np.uint8)
        
    img = Image.fromarray(sdf_viz).resize((data.shape[1] * scale, data.shape[0] * scale), Image.NEAREST)
    img.save(f"{name}_sdf.png")

def process_maps():
    map_names = ["park", "orchard", "cave", "forest"]
    circle_radii = [50, 70, 50, 70]
    
    for name, radius in zip(map_names, circle_radii):
        try:
            pgm_image = Image.open(f"{name}.pgm").convert("L")
            data = np.array(pgm_image)
            
            save_mission_map(data, name, 10, radius)
            save_traversability_map(data, name, 10)
            save_sdf_map(data, name, 10)
            
            print(f"Generated Figure 1, 2, and 3 for: {name}")
        except FileNotFoundError:
            print(f"Skipping {name}: file not found.")

if __name__ == "__main__":
    process_maps()