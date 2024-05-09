import cv2
import torch
from yolov5 import YOLOv5

def main():
    # Load the pre-trained YOLOv5s model
    model_path = 'runs/train/2024_03_17_img-640_batch-25_epochs-50/weights/best.pt'  # Adjust path as necessary
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = YOLOv5(model_path, device=device)

    # Initialize video capture from webcam
    cap = cv2.VideoCapture("E:/Coding/Pollen/datasets/SYNTH_POLEN23E_Manual_Filtered_300_100_filtered_FOR_INF/videos/val.avi")
    if not cap.isOpened():
        print("Error: Could not open video device.")
        return

    # Set webcam properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Main loop
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Perform inference
        results = model.predict(frame)
        results.render()  # Draw bounding boxes and labels on the frame

        # Count objects
        object_counts = {}
        for *xyxy, conf, cls in results.xyxy[0]:  # results.xyxy[0] is the tensor containing detections
            label = results.names[int(cls)]  # Get the label for each detection
            if label in object_counts:
                object_counts[label] += 1
            else:
                object_counts[label] = 1

        # Display object counts
        display_text = " | ".join(f"{key}: {value}" for key, value in object_counts.items())
        cv2.putText(frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Show the frame
        cv2.imshow("YOLOv5s Object Detection", frame)

        # Exit loop if ESC key is pressed
        if cv2.waitKey(1) == 27:
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
