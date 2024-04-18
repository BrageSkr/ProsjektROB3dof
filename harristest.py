import cv2
import numpy as np

# Parameters for Harris corner detection
harris_block_size = 2
harris_ksize = 3
harris_k = 0.1

# Parameters for corner tracking
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.03)
subpixel_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.01)

# Capture video from webcam
cap = cv2.VideoCapture(0)

# Initialize previous points for tracking
prev_points = None

while True:
    # Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Harris corner detection
    corners = cv2.cornerHarris(gray, harris_block_size, harris_ksize, harris_k)

    # Threshold corners to retain only strong corners
    threshold = 0.0000005 * corners.max()
    corner_mask = np.uint8(corners > threshold)


    # Find centroids of corners
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(corner_mask)

    if ret > 1:
        # Perform corner subpixel refinement
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        refined_corners = cv2.cornerSubPix(gray, np.float32(centroids), (5, 5), (-1, -1), criteria)

        # Draw refined corners on the frame
        for corner in refined_corners:
            x, y = corner.ravel()
            cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)

        # Track corners using Lucas-Kanade optical flow
        if prev_points is not None:
            new_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_points, None)

            # Select good points
            good_new = new_points[status == 1]
            good_prev = prev_points[status == 1]

            # Draw tracks
            for i, (new, prev) in enumerate(zip(good_new, good_prev)):
                x_new, y_new = new.ravel()
                x_prev, y_prev = prev.ravel()
                cv2.line(frame, (int(x_new), int(y_new)), (int(x_prev), int(y_prev)), (0, 255, 0), 2)
                cv2.circle(frame, (int(x_new), int(y_new)), 5, (0, 255, 0), -1)

        # Update previous points and frame
        prev_gray = gray.copy()
        prev_points = refined_corners.reshape(-1, 1, 2)

    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()

