# Importing libraries
import cv2

# Loading Image Path
imagePath = "img/1.png"

# Loading Haar Cascade (xml) Path
cascPath = "./haarcascade_frontalface_default.xml"

# Activating Haar Cascade (Face) Classifier
faceCascade = cv2.CascadeClassifier(cascPath)

# Reading the Image (Creating Object)

img = cv2.imread(imagePath)

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
