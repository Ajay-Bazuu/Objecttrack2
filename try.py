import cv2
import numpy as np

# Load the pre-trained model (SSD with MobileNet architecture)
model_prototxt = "deploy.prototxt"  # Path to your prototxt file
model_weights = "mobilenet_iter_73000.caffemodel"  # Path to your caffemodel file

# Initialize the list of class labels MobileNet SSD was trained to detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle",
           "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
           "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
           "train", "tvmonitor"]

# Load the pre-trained model and the class labels
net = cv2.dnn.readNetFromCaffe(model_prototxt, model_weights)

# Open the video capture from the web camera (index 0 is typically the default camera)
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Define the previous position of the bottle
prev_center = None

while True:
    # Capture frame-by-frame from the web camera
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame from webcam.")
        break

    # Get frame dimensions
    (h, w) = frame.shape[:2]

    # Prepare the frame for the model (resize and blob)
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1/255.0, (300, 300), (0, 0, 0))
    net.setInput(blob)
    detections = net.forward()

    bottle_center = None

    # Loop over the detections and draw boxes around objects
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:  # Confidence threshold
            idx = int(detections[0, 0, i, 1])
            if CLASSES[idx] == "bottle":
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # Compute the center of the bounding box
                bottle_center = ((startX + endX) // 2, (startY + endY) // 2)

                # Draw the bounding box and label on the frame
                label = f"{CLASSES[idx]}: {confidence:.2f}"
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # If a bottle was detected, track its movement
    if bottle_center:
        if prev_center:
            # Compute direction of movement
            dx = bottle_center[0] - prev_center[0]
            dy = bottle_center[1] - prev_center[1]

            if abs(dx) > abs(dy):
                if dx > 0:
                    direction_x = "right"
                else:
                    direction_x = "left"
                direction_y = ""
            else:
                if dy > 0:
                    direction_y = "down"
                else:
                    direction_y = "up"
                direction_x = ""

            # Display direction
            direction = f"Move: {direction_x} {direction_y}".strip()
            cv2.putText(frame, direction, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Update the previous center position
        prev_center = bottle_center

    # Display the output frame with bounding boxes and direction
    cv2.imshow("Webcam Object Detection", frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
