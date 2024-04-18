import cv2
import numpy as np
import matplotlib.pyplot as plt


# Function for edge detection
def edge_detection(frame):
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Parameters for edge detection
    dt = 0.1
    numstep = 1

    # Initialize arrays
    F = np.zeros_like(gray, dtype=float)
    F[:, :] = gray[:, :]
    Fx = np.zeros_like(F)
    Fy = np.zeros_like(F)
    Fxx = np.zeros_like(F)
    Fyy = np.zeros_like(F)

    # Perform edge detection
    for i in range(numstep):
        Fx[1:-1, :] = (F[2:, :] - F[:-2, :]) / 2.0
        Fy[:, 1:-1] = (F[:, 2:] - F[:, :-2]) / 2.0
        Fxx[1:-1, :] = (Fx[2:, :] - Fx[:-2, :]) / 2.0
        Fyy[:, 1:-1] = (Fy[:, 2:] - Fy[:, :-2]) / 2.0
        F[1:-1, 1:-1] = F[1:-1, 1:-1] + dt * (Fxx[1:-1, 1:-1] + Fyy[1:-1, 1:-1])

    # Calculate gradient magnitude
    Fgradabs = np.sqrt(Fx * Fx + Fy * Fy)
    minverdi = np.min(Fgradabs)
    maxverdi = np.max(Fgradabs)
    Fgradabsskalert = Fgradabs / (maxverdi - minverdi)

    # Thresholding function
    def kutt3(x):
        c = 0.04
        if x > c:
            return 1.0
        else:
            return 0.0

    # Apply thresholding function
    kuttnfunc = np.frompyfunc(kutt3, 1, 1)
    kanter = np.vectorize(kuttnfunc)(Fgradabsskalert)
    kanter = kanter.astype('float64')

    return kanter


# Initialize webcam capture
cap = cv2.VideoCapture(0)

# Main loop for processing frames
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Perform edge detection
    edges = edge_detection(frame)

    # Display the resulting frame
    cv2.imshow('Edge Detection', edges)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
