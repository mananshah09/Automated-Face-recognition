
from PIL import Image
from tflite_runtime.interpreter import Interpreter 
from tensorflow.python.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
import pickle
import cv2
# from matplotlib import pyplot as plt
from mtcnn.mtcnn import MTCNN
# from matplotlib.patches import Rectangle,Circle
import numpy as np


#  pip install --index-url https://google-coral.github.io/py-repo/ tflite_runtime

# print("Started level 0")
detector = MTCNN()
in_encoder = Normalizer(norm='l2')
# out_encoder = pickle.load(open("LabelEncoder.pickle", 'rb'))
svm_model = pickle.load(open("face_model15.pickle", 'rb'))
model_path ="facenet_model.tflite"
interpreter = Interpreter(model_path)
interpreter.allocate_tensors()
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
# print("Reached level 1")
classes_names = ["Arnav","Brendan","Jayesh","Manan","Prachi","Unknown"]

def get_embedding(face_pixels):
    # scale pixel values
    face_pixels = face_pixels.astype('float32')
    # standardize pixel values across channels (global)
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    # transform face into one sample
    samples = np.expand_dims(face_pixels, axis=0)
    # make prediction to get embedding
    
    interpreter.invoke()
    output_details = interpreter.get_output_details()[0]
    yhat = interpreter.get_tensor(output_details['index'])
    return yhat[0]

def set_input_tensor2(image):
  tensor_index = interpreter.get_input_details()[0]['index']
  # print("Index of the input tensor: ", tensor_index, end="\n\n")
  # Return the input tensor based on its index.
  input_tensor = interpreter.tensor(tensor_index)()[0]
  # Assigning the image to the input tensor.
  input_tensor[:, :] = image
  # print("inside set input tensor 4")


def set_input_tensor3(face_pixels):
  # scale pixel values
  face_pixels = np.asarray(face_pixels)
  face_pixels = face_pixels.astype(np.uint8)
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
  return scores

try:
    # print("Inside Level 2")
    while True:
        # Read the frame
        _, img = cap.read()

        # Convert to grayscale
    #     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('img', img)
        # Detect the faces
        faces = detector.detect_faces(img)
    #         plt.imshow(img)
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
        # print("Inside Level 3")
        for result in faces:
            (x, y, w, h) = result["box"]
            x, y = abs(x), abs(y)
            crop_img = img[y:y+h, x:x+w]
            crop_img = cv2.resize(crop_img, (160, 160))
            face_array = np.asarray(crop_img)
            set_input_tensor3(crop_img)
            # set_input_tensor(face_array)
            embedding = classify_image()
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
            print("{}) class:{}, Probability: {}".format(count,predict_names,probability_percent[yhat_class]))
            count+=1
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(img, predict_names, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
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
        # print("Inside Level 8")
except Exception as e:
    print(e)
    cap.release()
    cv2.destroyAllWindows()
#         plt.show()
