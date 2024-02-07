
# from PIL import Image
from tensorflow.python.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
import pickle
import cv2
# from matplotlib import pyplot as plt
# from mtcnn.mtcnn import MTCNN
# from matplotlib.patches import Rectangle,Circle
import numpy as np


# detector = MTCNN()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
in_encoder = Normalizer(norm='l2')
# out_encoder = pickle.load(open("LabelEncoder.pickle", 'rb'))
svm_model = pickle.load(open("face_model.pickle", 'rb'))
FaceNet_model = load_model('facenet_keras.h5')
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
classes_names = ["Arnav","Brendan","Jayesh","Manan","Prachi","Unknown"]
def get_embedding(model, face_pixels):
    # scale pixel values
    face_pixels = face_pixels.astype('float32')
    # standardize pixel values across channels (global)
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    # transform face into one sample
    samples = np.expand_dims(face_pixels, axis=0)
    # make prediction to get embedding
    yhat = model.predict(samples)
    return yhat[0]

try:
    while True:
        # Read the frame
        # cap = cv2.VideoCapture(0)

        _, img = cap.read()
        # cv2.imshow('img', img)
        # Convert to grayscale
    #     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect the faces
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        # plt.imshow(img)
    #         plt.xticks([]), plt.yticks([])  # Hides the graph ticks and x / y axis
    #         ax = plt.gca()
        # print(faces[0])
    #         pixels = np.asarray(img)
        # Draw the rectangle around each face
        if len(faces)==0:
            continue
        else:
            pass

        count = 1
        for (x, y, w, h) in faces:
            # x, y = abs(x), abs(y)
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            crop_img = img[y:y+h, x:x+w]
            crop_img = cv2.resize(crop_img, (160, 160))
            face_array = np.asarray(crop_img)
            embedding = get_embedding(FaceNet_model, face_array)
            testX = in_encoder.transform([embedding])
            random_face_emb = testX[0]
            samples = np.expand_dims(random_face_emb, axis=0)
            yhat_prob = svm_model.predict_proba(samples)
            yhat_class = np.argmax(yhat_prob)
            # print("class requires is :",yhat_class)
            predict_names = classes_names[yhat_class]
            probability_percent = yhat_prob[0]
            print("{}) class:{}, Probability: {}".format(count,predict_names,probability_percent[yhat_class]))
            count+=1
            cv2.putText(img, predict_names, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    #             rect = Rectangle((x, y), w, h, fill=False, color='red')
    #             ax.add_patch(rect)
    #             for key, value in result['keypoints'].items():
    #                 # create and draw dot
    #                 dot = Circle(value, radius=2, color='red')
    #                 ax.add_patch(dot)


        # Display
        # _, img = cap.read()
        cv2.imshow('img', img)
        cv2.waitKey(27)
    #         plt.show()    
except Exception as e:
    cap.release()
    cv2.destroyAllWindows()
