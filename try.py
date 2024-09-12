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

# Initialize the CSRT tracker
tracker = cv2.TrackerCSRT_create()
tracking = False

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

    if not tracking:
        # Loop over the detections and draw boxes around objects
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > 0.5:  # Confidence threshold
                idx = int(detections[0, 0, i, 1])
                if CLASSES[idx] == "person":
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    bbox = (startX, startY, endX - startX, endY - startY)

                    # Initialize the tracker with the detected bounding box
                    tracker.init(frame, bbox)
                    tracking = True
                    break

    else:
        # Update the tracker and get the new position
        success, bbox = tracker.update(frame)

        if success:
            # Draw the bounding box around the tracked object
            (x, y, w, h) = [int(v) for v in bbox]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Tracking Person", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            # Reset tracking if the object is lost
            tracking = False

    # Display the output frame with bounding boxes and tracking info
    cv2.imshow("Webcam Object Tracking", frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
