import cv2
import numpy as np
from keras.applications.resnet import ResNet101
from keras.applications import VGG16,MobileNet
from keras.models import Model
from keras.layers import Dense,Flatten,Dropout
import argparse
from beepy import beep
from time import time
def readFromImage(imgName):
    frame = cv2.imread(imgName)
    temp=frame.shape
    frame=cv2.resize(frame,(224, 224))
    frame = frame.reshape(1,224,224,3)
    pred=model.predict(frame)
        
    print(pred)
        # Display the resulting frame
    frame = frame.reshape(224,224,3)
    if(args.rotate):
        frame=cv2.rotate(frame,cv2.ROTATE_180)
    frame=cv2.resize(frame,(temp[1],temp[0]))
    frame=cv2.putText(frame, 'Alert : '+str(round(pred[0][0]*100,2)), (3,20), cv2.FONT_HERSHEY_PLAIN, 1,(0,0,255),2, cv2.LINE_AA)
    frame=cv2.putText(frame, 'Low Vigilant : '+str(round(pred[0][1]*100,2)), (3,40), cv2.FONT_HERSHEY_PLAIN, 1,(0,255,0),2, cv2.LINE_AA)
    frame=cv2.putText(frame, 'Drowsy : '+str(round(pred[0][2]*100,2)), (3,60), cv2.FONT_HERSHEY_PLAIN, 1,(255,0,0),2, cv2.LINE_AA)
    frame =cv2.rectangle(frame, (0,0), (200,70), (0,0,255), 2)
        
    cv2.imshow('Image', frame)
          
    cv2.waitKey(0)


def readFromVideo(vidName):
    vid = cv2.VideoCapture(vidName)
    while(True):
          
        # Capture the video frame
        # by frame
        start=time()
        ret, frame = vid.read()
        temp=frame.shape
        frame=cv2.resize(frame,(224, 224))
        frame = frame.reshape(1,224,224,3)
        pred=model.predict(frame)
        
        print(pred)
        # Display the resulting frame
        frame = frame.reshape(224,224,3)
        frame=cv2.resize(frame,(temp[1],temp[0]))
        if(args.rotate):
            frame=cv2.rotate(frame,cv2.ROTATE_180)
        frame=cv2.putText(frame, 'Alert : '+str(round(pred[0][0]*100,2)), (3,20), cv2.FONT_HERSHEY_PLAIN, 1,(0,0,255),2, cv2.LINE_AA)
        frame=cv2.putText(frame, 'Low Vigilant : '+str(round(pred[0][1]*100,2)), (3,40), cv2.FONT_HERSHEY_PLAIN, 1,(0,255,0),2, cv2.LINE_AA)
        frame=cv2.putText(frame, 'Drowsy : '+str(round(pred[0][2]*100,2)), (3,60), cv2.FONT_HERSHEY_PLAIN, 1,(255,0,0),2, cv2.LINE_AA)
        frame =cv2.rectangle(frame, (0,0), (200,70), (0,0,255), 2)
        end=time()
        print("Time to inference : "+str((end-start)*10))
        cv2.imshow('Webcam Feed', frame)
          
        if(np.argmax(pred[0])==2):
            beep(sound='ping')
        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
      
    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()

def getResnet101():
    vgg_model = ResNet101(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = vgg_model.output
    x = Flatten()(x) # Flatten dimensions to for use in FC layers
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x) # Dropout layer to reduce overfitting
    x = Dense(256, activation='relu')(x)
    x = Dense(3, activation='softmax')(x) # Softmax for multiclass
    transfer_model = Model(inputs=vgg_model.input, outputs=x)
    transfer_model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])
    transfer_model.load_weights('resnet101.h5')
    return transfer_model


def getMobileNet():
    vgg_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = vgg_model.output
    x = Flatten()(x) # Flatten dimensions to for use in FC layers
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x) # Dropout layer to reduce overfitting
    x = Dense(256, activation='relu')(x)
    x = Dense(3, activation='softmax')(x) # Softmax for multiclass
    transfer_model = Model(inputs=vgg_model.input, outputs=x)
    transfer_model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])
    transfer_model.load_weights('mobile.h5')
    return transfer_model
    
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", type=str, default='0',help = "Input 0 for webcam and file name for a video/image")
parser.add_argument("-t", "--type", type=str, default='v',help = "Read from Video(v) or Image(i)")
parser.add_argument("-r", "--rotate", type=bool, default=False,help = "Rotate image by 180 or not")
parser.add_argument("-m", "--model", type=str, default='Resnet101',choices=['Resnet101','MobileNet'],help = "Which model to use")
args = parser.parse_args()

if(args.model=='MobileNet'):
    model=getMobileNet()
else:
    model=getResnet101()

if(args.type=='i'):
    readFromImage(args.input)
else:
    if(args.input=='0'):
        args.input=0
    readFromVideo(args.input)

