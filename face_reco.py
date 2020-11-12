import cv2
import sys

# Get user supplied values
cascPath="haarcascade.xml"

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)
path=sys.argv[1]

# Read the image
image = cv2.imread(path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=6,
    minSize=(30, 30),
    #flags = cv2.cv.CV_HAAR_SCALE_IMAGE
)

print("Found {0} faces!".format(len(faces)))

for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)



# Export the result
# cv2.imwrite("face_detected.png", image) 
# print('Successfully saved')

cv2.imshow("Faces found", image)
cv2.waitKey(0)
