import cv2


# Function for edge detection
def edge_detection(frame):
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 1)

    # Gradient calculation
    grad_x = cv2.Sobel(blurred, cv2.CV_64F, 2, 0, ksize=5)
    grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 2, ksize=5)
    gradient = cv2.magnitude(grad_x, grad_y)

    # Thresholding
    _, edges = cv2.threshold(gradient, 100, 250, cv2.THRESH_BINARY)

    return edges


# Initialize webcam capture
cap = cv2.VideoCapture(0)

# Main loop for processing frames
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Resize frame for faster processing
    resized_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

    # Perform edge detection
    edges = edge_detection(resized_frame)

    # Display the resulting frame
    cv2.imshow('Edge Detection', edges)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()

