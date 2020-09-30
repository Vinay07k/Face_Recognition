# Importing libraries
import cv2

def recognize_face(img_path):
    # Loading Haar Cascade (xml) Path
    cascPath = "./haarcascade_frontalface_default.xml"

    # Activating Haar Cascade (Face) Classifier
    faceCascade = cv2.CascadeClassifier(cascPath)

    # Reading the Image (Creating Object)

    img = cv2.imread(img_path)

    # Creating a Grayscale image
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detecting faces in the grayscale image
    scaleFactor = 1.2
    minNeighbors = 5
    faces = faceCascade.detectMultiScale(grayImg, scaleFactor, minNeighbors)

    print(f"Number of Faces: {len(faces)}")

    # Drawing a Red Rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0,0,255), 2)

    # Displaying the image and waiting for a key to exit
    cv2.imshow("FACES", img)
    cv2.waitKey()
    cv2.destroyAllWindows()

def capture_photo():
    cap = VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if ret:
            cv2.imshow("Frame", frame)
            cv2.imwrite("capture.jpg", frame)

            if cv2.waitKey(1) & 0xFF == 27: # Press Escape to Stop
                break
        else:
            print("Unable to load camera!")

    cap.release()
    cv2.destroyAllWindows()
