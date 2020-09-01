# Importing Libraries
import cv2

# Loading Harr Cascade (xml) Path
cascPath = "./haarcascade_frontalface_default.xml"

# Activating Haar Cascade (Face) Classifer
faceCascade = cv2.CascadeClassifier(cascPath)

# Starting Webcam Capture
cap = cv2.VideoCapture(0)

while cap.isOpened():

    # Reading Frame of the Capture
    ret, frame = cap.read()

    # Creating a GrayScale image
    grayImg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detecting faces in grayscale image
    scaleFactor = 1.2
    minNeighbors = 5
    faces = faceCascade.detectMultiScale(grayImg, scaleFactor, minNeighbors)

    # Drawing a Red Rectangle around faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,0,255), 2)

    # Displaying the frame
    cv2.imshow('Image', frame)

    # Waiting for 'q' to be pressed at 1 millisecond
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
