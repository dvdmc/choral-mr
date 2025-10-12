import numpy as np
import matplotlib.pyplot as plt
from queue import Queue

TRAVERSABILITY_VALUE = 51
FREE_THRESH = 90
DIST_THRESH = 0.1
RESOLUTION = 0.0333  # meters per pixel

def load_pgm(filename):
    with open(filename, 'r') as f:
        assert f.readline().strip() == 'P2'
        line = f.readline().strip()
        while line.startswith('#'):
            line = f.readline().strip()

        width, height = map(int, line.split())
        maxval = int(f.readline().strip())
        data = []
        for line in f:
            data.extend(map(int, line.split()))
        image = np.array(data).reshape((height, width))
        image = np.flipud(image)  # Flip to match bottom-left origin
        return image

def generate_grid_map(pgm_data):
    grid_map = 100 - (pgm_data * 100) // 255  # Invert and scale 0-255 to 100-0
    return grid_map

def compute_traversability_map(grid_map):
    grid_map_terrain = (grid_map == TRAVERSABILITY_VALUE).astype(np.uint8)
    return grid_map_terrain

def compute_sdf(grid_map):
    h, w = grid_map.shape
    obstacle_mask = (grid_map >= FREE_THRESH).astype(np.uint8)

    dist = np.full((h, w), np.inf, dtype=np.float32)
    dist[obstacle_mask == 1] = 0

    q = Queue()
    for r in range(h):
        for c in range(w):
            if obstacle_mask[r, c] == 1:
                q.put((r, c))

    directions = [(-1,0),(1,0),(0,-1),(0,1)]
    while not q.empty():
        r, c = q.get()
        for dr, dc in directions:
            nr, nc = r+dr, c+dc
            if 0 <= nr < h and 0 <= nc < w:
                if dist[nr, nc] > dist[r, c] + 1:
                    dist[nr, nc] = dist[r, c] + 1
                    q.put((nr, nc))

    return dist * RESOLUTION

def visualize_traversability(grid_map, terrain_map):
    h, w = grid_map.shape
    img = np.ones((h, w, 3), dtype=np.float32)  # Start with white

    for r in range(h):
        for c in range(w):
            if grid_map[r, c] >= FREE_THRESH:
                img[r, c] = [0.0, 0.0, 0.0]  # Black for obstacles
            elif terrain_map[r, c] == 1:
                img[r, c] = [0.6, 0.3, 0.0]  # Brown for bad terrain

    plt.figure(figsize=(10, 10))
    plt.imshow(img, origin='lower')
    plt.title("Traversability Map")
    plt.axis('off')
    plt.show()

def visualize_sdf(sdf):
    plt.figure(figsize=(10, 10))
    plt.imshow(sdf, cmap='viridis', origin='lower')
    plt.colorbar(label="Distance (m)")
    plt.title("Signed Distance Field (SDF)")
    plt.axis('off')
    plt.show()

# ---------- USAGE EXAMPLE -------------
filename = '/root/mr_ws/src/het_vrp/maps/cave.pgm'
pgm = load_pgm(filename)
grid_map = generate_grid_map(pgm)
terrain_map = compute_traversability_map(grid_map)
sdf = compute_sdf(grid_map)

visualize_traversability(grid_map, terrain_map)
visualize_sdf(sdf)