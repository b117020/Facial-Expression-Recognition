import cv2
from model import FacialExpressionModel
import numpy as np

facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = FacialExpressionModel("model1.json", "model_weights.h5")
video = cv2.VideoCapture('videos/prabhat.webm')
predictions=[]

def get_frame():
        try:
            _, fr = video.read()
            gray_fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            faces = facec.detectMultiScale(gray_fr, 1.3, 5)
            preds=[]
        except:
            return 
        
        for (x, y, w, h) in faces:
            fc = gray_fr[y:y+h, x:x+w]

            roi = cv2.resize(fc, (48, 48))
            pred = model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])
            preds.append(pred)
            
        return preds
    
fcount = 0  
while True:
    try:
        preds = get_frame()
        fcount += 1
        
        if(len(preds) != 0):
            for item in preds:
                predictions.append(item)
            print(preds)
    except:
        break


positivity=0
negativity=0
for item in predictions:
    if item in ["Angry", "Disgust", "Sad", "Fear"]:
        negativity += 1
    else:
        positivity += 1
        
percent_positivity = positivity / fcount * 100
percent_negativity = negativity / fcount * 100