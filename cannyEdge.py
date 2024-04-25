import cv2
import numpy as np

# Function for ball detection
def detect_balls(frame):
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 1)
    edges = cv2.Canny(blurred, 10, 200)

    # Find circles using Hough Circle Transform
    detected_circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 10, param1=50, param2=30, minRadius=40, maxRadius=80)

    # Draw circles on the frame
    if detected_circles is not None:
        detected_circles = np.round(detected_circles[0, :]).astype("int")
        for (x, y, r) in detected_circles:
            print(x,y,r)
            cv2.circle(edges, (x, y), r, (255, 255, 0), 10)

    return edges

# Initialize webcam capture
cap = cv2.VideoCapture(0)

# Main loop for processing frames
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Perform ball detection
    result_frame = detect_balls(frame)

    # Display the resulting frame
    cv2.imshow('Ball Detection', result_frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
