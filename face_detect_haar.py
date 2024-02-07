import cv2
# from matplotlib import pyplot as plt

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

try:
    while True:
        # Read the frame
        _, img = cap.read()

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect the faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        # plt.imshow(gray)
        # plt.xticks([]), plt.yticks([])  # Hides the graph ticks and x / y axis
        # plt.show()
        print(faces)
        # Draw the rectangle around each face
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Display
        cv2.imshow('img', img)

        # Stop if escape key is pressed
        k = cv2.waitKey(30) & 0xff
        if k==27:
            break
except Exception as e:
    # Release the VideoCapture object
    cap.release()


        
