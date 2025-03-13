import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from pathlib import Path
import argparse



def vis_detections(img, bboxes, output_path, xywh=True):
    if not xywh:
        bboxes = [(x0, y0, x1 - x0, y1 - y0) for x0, y0, x1, y1 in bboxes]

    colors = ["red", "blue", "green", "yellow", "purple", "orange", "brown", "pink", "gray", "cyan"]
    fig,ax = plt.subplots(1)
    ax.imshow(img)

    for idx, bbox in enumerate(bboxes):
        x, y, w, h = bbox
        color = colors[idx % len(colors)]
        rect = patches.Rectangle((x, y), w, h, linewidth=2, facecolor='none', edgecolor=color)
        ax.add_patch(rect)
        ax.text(x+10, y + 40, f"{idx}", color="white", fontsize=12, bbox=dict(facecolor=color, alpha=0.5))

    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()


def visualize(video, proposals):
    # Define paths
    data_dir = Path("data")
    image_dir = data_dir / "datasets" / "videos" / video
    json_file = data_dir / "results" / "videos" / video / proposals

    # List all the files in the image directory, sort them, and pick the first file
    image_files = sorted(image_dir.iterdir())
    if len(image_files) == 0:
        raise FileNotFoundError("No image files found in the specified directory.")
    image_file = image_files[0]

    # Load the image
    img = Image.open(image_file)

    # Load and parse the JSON file
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Filter objects by 'image_id' == 0
    bboxes = [obj["bbox"] for obj in data if obj.get('image_id') == 0]

    # Draw bounding boxes on the image
    output_path = data_dir / "results" / "videos" / video / f"viz_detections_{video}.png"
    vis_detections(img, bboxes, output_path)
    print(f"Saved visualization to {output_path.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize objects in an image with bounding boxes.')
    parser.add_argument('--video', type=str, required=True, help='Name of the video folder.')
    parser.add_argument('--proposals', type=str, required=True)
    args = parser.parse_args()

    visualize(args.video, args.proposals)
