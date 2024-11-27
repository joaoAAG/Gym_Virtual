import torch
from ultralytics import YOLO


def load_model(weights_path):
    # Load the YOLO model with the specified weights
    model = YOLO(weights_path)
    return model


def perform_inference(model, image_path):
    # Perform inference on a given image
    results = model(image_path)
    return results


if __name__ == '__main__':
    weights_path = r'C:\Users\joaoa\PycharmProjects\SiaviFitV4_Exercises\runs\face-recognition6\weights\best.pt'
    image_path = r'paulo (187).jpg'  # Replace with the path to your test image

    # Load the model
    model = load_model(weights_path)

    # Perform inference
    results = perform_inference(model, image_path)

    # Print and visualize the results
    for result in results:
        print(result.verbose())  # Print the results
        result.show()  # Display the results
