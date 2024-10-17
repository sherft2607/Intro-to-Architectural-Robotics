import cv2
import numpy as np

# Capture video from the default camera
cap = cv2.VideoCapture(0)  # Use 0 for the default camera, or adjust as needed

# RGB value to focus on (60, 108, 139)
target_color_rgb = (60, 108, 139)

# Convert target RGB to HSV
target_color_hsv = cv2.cvtColor(np.uint8([[target_color_rgb]]), cv2.COLOR_RGB2HSV)[0][0]

# Define a larger range for the blue color in HSV
lower_blue = np.array([target_color_hsv[0] - 20, 50, 50])
upper_blue = np.array([target_color_hsv[0] + 20, 255, 255])

# Threshold area to filter out small contours
threshold_area = 100

while True:
    # Capture a frame from the video feed
    ret, frame = cap.read()

    # Convert the frame to the HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a mask for the blue color
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    # Bitwise-AND to extract the blue-colored regions
    result = cv2.bitwise_and(frame, frame, mask=mask_blue)

    # Object Detection: Find contours in the blue mask
    contours, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw rectangles around the contours that approximate rectangles
    for contour in contours:
        if cv2.contourArea(contour) > threshold_area:
            # Calculate area
            area = cv2.contourArea(contour)

            # Calculate centroid
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                # Draw the contour, rectangle, and centroid on the frame
                cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

                # Display area and centroid coordinates
                cv2.putText(frame, f"Area: {area}", (cx - 50, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Display the original frame, color detection result, and object detection result
    cv2.imshow('Original Frame', frame)
    cv2.imshow('Blue Color Detection Result', result)

    # Handle user input to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
