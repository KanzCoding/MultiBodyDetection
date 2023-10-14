import cv2

# Create a CascadeClassifier for body detection
body_classifier = cv2.CascadeClassifier('haarcascade_fullbody.xml')  # Replace 'path_to_cascade_classifier.xml' with the actual path to your classifier file.

# Create a VideoCapture object to capture the video feed (you can replace '0' with the path to a video file)
cap = cv2.VideoCapture('walking.avi')

while True:
    # Read a frame from the video capture
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Use the body_classifier to detect bodies in the frame
    bodies = body_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Iterate over the detected bodies and draw rectangles around them
    for (x, y, w, h) in bodies:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

    # Display the frame with detected bodies
    cv2.imshow('Body Detection', frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
