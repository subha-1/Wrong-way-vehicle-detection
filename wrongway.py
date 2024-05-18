
import cv2
import numpy as np

def detect_moving_vehicles(video_path):
    cap = cv2.VideoCapture(sibani_video_(1).mp4)
    ret, frame1 = cap.read()
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    while True:
        ret, frame2 = cap.read()
        if not ret:
            break

        next_frame = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # Calculate dense optical flow
        flow = cv2.calcOpticalFlowFarneback(prvs, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # Get magnitude and direction of flow
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # Threshold to identify moving pixels
        threshold = 20
        moving_pixels = (mag > threshold).astype(np.uint8)

        # Find contours of moving objects
        contours, _ = cv2.findContours(moving_pixels, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw rectangles around moving vehicles
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            color = (0, 255, 0)  # Green rectangle for right direction
            if flow[y + h // 2, x + w // 2, 1] < 0:
                color = (0, 0, 255)  # Blue rectangle for wrong direction

            cv2.rectangle(frame2, (x, y), (x + w, y + h), color, 2)
        # Display the result
        cv2.imshow('Moving Vehicles Detection', frame2)
        if cv2.waitKey(30) & 0xFF == 13:
            break

        prvs = next_frame

    cap.release()
    cv2.destroyAllWindows()

# Replace 'your_video_path.mp4' with the path to your video file
video_path = 'video (1).mp4'
detect_moving_vehicles(video_path)
