import cv2
import torch
import math
from yolov5 import YOLOv5

def main():
    # Load the pre-trained YOLOv5s model
    model_path = 'runs/train/2024_03_17_img-640_batch-25_epochs-50/weights/best.pt'  # Adjust path as necessary
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = YOLOv5(model_path, device=device)

    # Initialize video capture
    cap = cv2.VideoCapture("E:/Coding/Pollen/datasets/SYNTH_POLEN23E_Manual_Filtered_300_100_filtered_FOR_INF/videos/val.avi")
    if not cap.isOpened():
        print("Error: Could not open video device.")
        return

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter('out.avi', fourcc, 30.0, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    # Cumulative counts dictionary
    cumulative_counts = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Perform inference
        results = model.predict(frame)
        results.render()  # Draw bounding boxes and labels on the frame

        # Update counts for current frame
        frame_counts = {}
        for *xyxy, conf, cls in results.xyxy[0]:
            label = results.names[int(cls)]
            frame_counts[label] = frame_counts.get(label, 0) + 1
            cumulative_counts[label] = cumulative_counts.get(label, 0) + 1

        # Display cumulative counts as a column
        start_y = 30
        for key, value in cumulative_counts.items():
            cv2.putText(frame, f"{key}: {math.ceil(value / (1920 / 10))}", (10, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)  # Pseudo Counter
            start_y += 30  # Move to the next line for each new key-value pair

        # Write the frame
        out.write(frame)

        # Show the frame
        cv2.imshow("YOLOv5s Pollen Detection", frame)

        if cv2.waitKey(1) == 27:
            break

    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
