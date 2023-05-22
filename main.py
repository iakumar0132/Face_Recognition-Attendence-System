import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = 'Training_images'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

# Compute face encodings for known images
knownEncodings = []
knownNames = []
for image, name in zip(images, classNames):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb, model='hog')
    encodings = face_recognition.face_encodings(rgb, boxes)
    for encoding in encodings:
        knownEncodings.append(encoding)
        knownNames.append(name)

print('Encoding Complete')

cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the frame from BGR to RGB for face recognition
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Find all the faces and face encodings in the current frame of video
    boxes = face_recognition.face_locations(rgb, model='hog')
    encodings = face_recognition.face_encodings(rgb, boxes)

    # Loop over the face detections
    for box, encoding in zip(boxes, encodings):
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(knownEncodings, encoding)
        name = "NOT_FOUND"  # Default name if face is not recognized

        # Find the best match
        faceDistances = face_recognition.face_distance(knownEncodings, encoding)
        bestMatchIndex = np.argmin(faceDistances)
        if matches[bestMatchIndex]:
            name = knownNames[bestMatchIndex]

        # Draw a box around the face
        top, right, bottom, left = box
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Draw a label with the name below the face
        cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

        # Mark attendance if the person is recognized
        if name != "NOT_FOUND":
            with open('Attendance.csv', 'r+') as f:
                attendanceData = f.read()
                if name not in attendanceData:
                    now = datetime.now()
                    dtString = now.strftime('%H:%M:%S')
                    f.write(f'\n{name},{dtString}')

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Quit the program when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()

