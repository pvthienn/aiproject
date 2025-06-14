from ultralytics import YOLO

if __name__ == '__main__':
    # Load a YOLO model
    model = YOLO("yolo11n.pt")

    # Train and Fine-Tune the Model
    model.train(data="data.yaml", epochs=500,            # number of epochs
                imgsz=640,             # image size for training
                batch=16,              # batch size, adjust based on GPU memory
                device=0,             # set device to 0 for a single GPU; set to 'cpu' to use CPU
                workers=1,            # number of workers for data loading
                project="YOLOv11_custom", # name of the project folder
                name="run1",           # experiment name
                save=True             # save checkpoints
    )