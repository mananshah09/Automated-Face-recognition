# from PIL import Image
from tflite_runtime.interpreter import Interpreter 
# from tensorflow.python.keras.models import load_model
# from sklearn.preprocessing import LabelEncoder
# import RPi.GPIO as GPIO
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
import pickle
import cv2
import numpy.ma as ma
import time
# from matplotlib import pyplot as plt
# from mtcnn.mtcnn import MTCNN
# from matplotlib.patches import Rectangle,Circle
import numpy as np


#  pip install --index-url https://google-coral.github.io/py-repo/ tflite_runtime

# print("Started level 0")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
in_encoder = Normalizer(norm='l2')
# out_encoder = pickle.load(open("LabelEncoder.pickle", 'rb'))
svm_model = pickle.load(open("face_model15.pickle", 'rb'))
model_path ="facenet_model.tflite"
interpreter = Interpreter(model_path)
interpreter.allocate_tensors()
# GPIO.setmode(GPIO.BCM)
# GPIO.setup(17, GPIO.OUT)
# Servo = GPIO.PWM(17,50)


# GPIO.setup(switch, GPIO.IN)

# print("Reached level 1")
classes_names = ["Arnav","Brendan","Jayesh","Manan","Prachi","Unknown"]

def servo_angle(angle):
    angle = round(angle*(11/180),1)+1.5
    # angle = round(angle*(-11/180),1)+12.5   ## for reverse direction of servo
    # Servo.ChangeDutyCycle(angle)
    time.sleep(1)

def set_input_tensor3(face_pixels):
  # scale pixel values
  #face_pixels = np.asarray(face_pixels)
  #face_pixels = face_pixels.astype(np.uint8)
    # standardize pixel values across channels (global)
  mean, std = face_pixels.mean(), face_pixels.std()
  face_pixels = (face_pixels - mean) / std
    # transform face into one sample
  samples = np.expand_dims(face_pixels, axis=0)
    # make prediction to get embedding
  tensor_index = interpreter.get_input_details()[0]['index']
  # print("Index of the input tensor: ", tensor_index, end="\n\n")
  # Return the input tensor based on its index.
  # print(samples.shape)
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = samples

def classify_image():
  # Call the invoke() method from inside a function to avoid this RuntimeError: reference to internal data in the interpreter in the form of a numpy array or slice.
  interpreter.invoke()
  output_details = interpreter.get_output_details()[0]
  # print("Details about the input tensors:\n   ", output_details, end="\n\n")
  scores = interpreter.get_tensor(output_details['index'])
  # scores = np.asarray(crop_img,dtype="float32")
  return scores

count_frames = 1
try:
    # print("Inside Level 2")
    curr_name = ""
    counter = [0,0]
    while True:
        # Read the frame
        print(count_frames)
        _, img = cap.read()
        # print(img)
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        # cv2.imshow('img', img)
        # Detect the faces
        # faces = detector.detect_faces(img)
    #         plt.imshow(img)
    #         plt.xticks([]), plt.yticks([])  # Hides the graph ticks and x / y axis
    #         ax = plt.gca()
        # print(faces[0])
    #         pixels = np.asarray(img)
        # Draw the rectangle around each face
        #cv2.imshow('img', img)
        if len(faces)==0:
            pass
        else:
            pass

        count = 1
        # print("Inside Level 3")
        for (x, y, w, h) in faces:
            # x, y = abs(x), abs(y)
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            crop_img = img[y:y+h, x:x+w]
            crop_img = cv2.resize(crop_img, (160, 160))
            # face_array = np.asarray(crop_img,dtype="int")
            set_input_tensor3(crop_img)
            # set_input_tensor(face_array)
            embedding = classify_image()
            # embedding = np.asarray(embedding,dtype="float32")
            embedding = np.where(np.isnan(embedding), ma.array(embedding, mask=np.isnan(embedding)).mean(axis=0), embedding)
            #print(embedding.shape)
            #print(np.any(np.isnan(embedding)))
            #print(np.all(np.isfinite(embedding)))
            # print("Inside Level 5")
            testX = in_encoder.transform(embedding)
            # print("Inside Level 5.1")
            random_face_emb = testX[0]
            # print("Inside Level 5.2")
            samples = np.expand_dims(random_face_emb, axis=0)
            # print("Inside Level 5.3")
            yhat_prob = svm_model.predict_proba(samples)
            # print("Inside Level 5.4")
            yhat_class = np.argmax(yhat_prob)
            # print("Inside Level 6")
            # print("class requires is :",yhat_class)
            predict_names = classes_names[yhat_class]
            probability_percent = yhat_prob[0]
            if count==1 and predict_names!="Unknown" and curr_name=="":
                curr_name = predict_names
                counter[0] = 1
                counter[1] = 1
            elif count==1 and predict_names==curr_name and counter[1]<15 :
                counter[0]+=1
                counter[1]+=1
            elif count==1 and predict_names!=curr_name and counter[1]<15:
                counter[1]+=1
            elif count==1 and counter[1]==15 and predict_names==curr_name:
                if counter[0]>=15:
                    print("Access Granted to {}".format(predict_names))
                    servo_angle(90)
                    servo_angle(0)
                counter=[0,0]
            elif counter[1]==15:
                counter = [0,0]
                curr_name = ""
            print("{}) class:{}, Probability: {}".format(count,predict_names,probability_percent[yhat_class]))
            
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(img,str(count)+") "+predict_names, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
            count+=1
            # print("Inside Level 7")
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
        count_frames+=1
        # print("Inside Level 8")
except Exception as e:
    print(e)
    
finally:    
    # GPIO.cleanup()
    cap.release()
    cv2.destroyAllWindows()
    
#         plt.show()