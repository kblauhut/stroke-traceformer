import svgwrite
import cairosvg
import random
import os
import json

# Constants
WIDTH, HEIGHT = 100, 100  # Canvas size
NUM_CURVES = 2  # Number of connected Bézier curves
STROKE_WIDTH = 2  # Thickness of the curve
MAX_STEP = 80  # Max movement per control point (limits overlap)
SMOOTHNESS_FACTOR = 0.6  # How much the next curve follows the previous direction
NUM_IMAGES = 200  # Number of images to generate
DATA_DIR = "data"  # Output directory

# Ensure the data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# Delete all files in data/ except .gitkeep
for filename in os.listdir(DATA_DIR):
    file_path = os.path.join(DATA_DIR, filename)
    if filename != ".gitkeep": # TODO Get this to delete folders as well
        os.remove(file_path)

def bounded_random_point(base, max_step=MAX_STEP):
    """Generate a point near the base but keep it inside the image boundaries."""
    x = min(max(base[0] + random.randint(-max_step, max_step), 10), WIDTH - 10)
    y = min(max(base[1] + random.randint(-max_step, max_step), 10), HEIGHT - 10)
    return (x, y)

def get_smooth_control_points(start, prev_control, smoothness=SMOOTHNESS_FACTOR):
    """Generate control points that follow the previous direction to ensure smoothness."""
    dx = start[0] - prev_control[0]
    dy = start[1] - prev_control[1]

    p1 = (start[0] + dx * smoothness, start[1] + dy * smoothness)
    p1 = bounded_random_point(p1)  # Add slight randomness
    p2 = bounded_random_point(p1)

    return p1, p2

def generate_smooth_connected_bezier_curves(file_id):
    """Create an SVG with a natural-looking, smoothly connected Bézier path."""
    folder_name = f"{DATA_DIR}/{file_id}"
    svg_filename = f"{folder_name}/curves.svg"
    png_filename = f"{folder_name}/curves.png"

    os.makedirs(folder_name, exist_ok=True)
    dwg = svgwrite.Drawing(svg_filename, size=(WIDTH, HEIGHT))

    start_point = (random.randint(WIDTH//4, 3*WIDTH//4), random.randint(HEIGHT//4, 3*HEIGHT//4))
    prev_control = bounded_random_point(start_point)
    path_data = f"M {start_point[0]},{start_point[1]}"
    json_path_data = [['M', start_point[0], start_point[1]]]

    for _ in range(NUM_CURVES):
        p1, p2 = get_smooth_control_points(start_point, prev_control)
        end_point = bounded_random_point(p2)

        if random.random() < 0.1:
            p1 = bounded_random_point(start_point, max_step=round(MAX_STEP * 1.5))

        path_data += f" C {p1[0]},{p1[1]}, {p2[0]},{p2[1]}, {end_point[0]},{end_point[1]}"
        json_path_data.append(['C', p1[0], p1[1], p2[0], p2[1], end_point[0], end_point[1]])
        prev_control = p2
        start_point = end_point

    dwg.add(dwg.path(d=path_data, stroke="black", fill="none", stroke_width=STROKE_WIDTH))
    dwg.save()

    # Write JSON file
    with open(f"{folder_name}/curves.json", "w") as f:
        json_str = json.dumps(json_path_data)
        f.write(json_str)

    # Convert to PNG
    cairosvg.svg2png(url=svg_filename, write_to=png_filename, dpi=300)

# Generate 100 images
for i in range(1, NUM_IMAGES + 1):
    generate_smooth_connected_bezier_curves(str(i).zfill(3))

print(f"Generated {NUM_IMAGES} images in the 'data/' folder.")
