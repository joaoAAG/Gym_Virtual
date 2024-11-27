import cv2
import os

# Set the path to your images and annotations directories
image_dir = r"training_images/train/images"
annotations_dir = r"training_images/train/labels"

os.makedirs(annotations_dir, exist_ok=True)

# Initialize global variables
ix, iy = -1, -1
drawing = False
img = None
img_copy = None

# Define the class names and their corresponding indices
class_names = ["Joao", "Paulo", "Jorge"]
class_indices = {name.lower(): index for index, name in enumerate(class_names)}

# Mouse callback function to draw rectangle
def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, img, img_copy

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            img = img_copy.copy()
            cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 2)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 2)
        # Calculate normalized coordinates
        x_center = (ix + x) / 2 / img.shape[1]
        y_center = (iy + y) / 2 / img.shape[0]
        width = abs(x - ix) / img.shape[1]
        height = abs(y - iy) / img.shape[0]
        # Save annotation in YOLO format with the pre-selected class index
        annotation = f"{selected_class_index} {x_center} {y_center} {width} {height}\n"
        with open(os.path.join(annotations_dir, f"{os.path.splitext(image_name)[0]}.txt"), "a") as f:
            f.write(annotation)

# Read and annotate images
for image_name in os.listdir(image_dir):
    if image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):  # Ensure you're only processing image files
        img = cv2.imread(os.path.join(image_dir, image_name))
        img_copy = img.copy()
        cv2.namedWindow("image")
        cv2.setMouseCallback("image", draw_rectangle)

        # Automatically determine the class index from the filename
        if 'joao' in image_name.lower():
            selected_class_index = class_indices['joao']
        elif 'paulo' in image_name.lower():
            selected_class_index = class_indices['paulo']
        elif 'jorge' in image_name.lower():
            selected_class_index = class_indices['jorge']
        else:
            print(f"Skipping {image_name}: class not recognized")
            continue

        print(f"Annotating {image_name} as {class_names[selected_class_index]}. Press 's' to save, 'c' to clear, 'q' to quit.")
        while True:
            cv2.imshow("image", img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                break
            elif key == ord('c'):
                img = img_copy.copy()

        cv2.destroyAllWindows()
