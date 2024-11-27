import torch
from ultralytics import YOLO

def main():
    # Check for GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    if device == 'cuda':
        print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
    else:
        print("CUDA is not available. Using CPU.")

    # Load the YOLOv8 model
    model = YOLO('yolov8s.yaml').to(device)

    # Define the dataset configuration path
    data_config = r'C:\Users\joaoa\PycharmProjects\SiaviFitV4_Exercises\faces_dataset.yaml'

    # Train the model with optimized settings
    model.train(
        data=data_config,
        epochs=479,                # Adjust as needed
        batch=32,                 # Batch size; adjust based on GPU memory
        imgsz=640,                # Image size
        device=device,            # Use GPU if available
        optimizer='AdamW',        # Faster optimizer
        amp=True,                 # Enable mixed precision
        workers=8,                # Number of workers for data loading
        save=True,                # Save model checkpoints
        project=r'C:\Users\joaoa\PycharmProjects\SiaviFitV4_Exercises\runs',    # Directory to save training logs
        name='face-recognition5'
    )

    print("Training completed!")

if __name__ == '__main__':
    main()
