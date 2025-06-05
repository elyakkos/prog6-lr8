import cv2
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageTk
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--image')
args = parser.parse_args()

tk = Tk()

tk.wm_title("HDetector v0.1")
tk.geometry('2560x1600')
tk.config(background="#FFFFFF")

tk.resizable(0, 0)

imageFrame = ttk.Frame(tk, width=2560, height=1600)
imageFrame.grid(row=0, column=0, padx=2, pady=2)

lmain = ttk.Label(imageFrame)
lmain.grid(row=0, column=0)
cap = cv2.VideoCapture(args.image if args.image else 0)

faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
genderProto="gender_deploy.prototxt"
genderModel="gender_net.caffemodel"
ageProto="age_deploy.prototxt"
ageModel="age_net.caffemodel"

MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
genderList=['Male ','Female']
ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

faceNet = cv2.dnn.readNet(faceModel, faceProto)
genderNet=cv2.dnn.readNet(genderModel,genderProto)
ageNet=cv2.dnn.readNet(ageModel,ageProto)


def highlightFace(net, frame, conf_threshold=0.4):
    frameOpencvDnn = frame.copy()

    frameHeight = frameOpencvDnn.shape[0]
    frameWIdth = frameOpencvDnn.shape[1]

    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)

    detections = net.forward()

    faceBoxes = []

    for i in range(detections.shape[2]):
        confidence = detections[0,0,i,2]
        if confidence >= conf_threshold:
            x1 = int(detections[0,0,i,3]*frameWIdth)
            y1 = int(detections[0,0,i,4]*frameHeight)

            x2 = int(detections[0,0,i,5]*frameWIdth)
            y2 = int(detections[0,0,i,6]*frameHeight)

            faceBoxes.append([x1, y1, x2, y2])

            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 0)
    
    return frameOpencvDnn, faceBoxes


def show_frame():
    hasFrame, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if hasFrame:
        resultImg, faceBoxes = highlightFace(faceNet, frame)

        for faceBox in faceBoxes:
            face=frame[max(0,faceBox[1]):
                min(faceBox[3],frame.shape[0]-1),max(0,faceBox[0])
                :min(faceBox[2], frame.shape[1]-1)]
            blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
            genderNet.setInput(blob)
            genderPreds=genderNet.forward()
            gender=genderList[genderPreds[0].argmax()]

            ageNet.setInput(blob)
            agePreds=ageNet.forward()
            age=ageList[agePreds[0].argmax()]

            cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1, cv2.LINE_AA)
        cv2image = cv2.cvtColor(resultImg, cv2.COLOR_BGR2RGBA)

        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)

        lmain.imgtk = imgtk
        lmain.configure(image=imgtk)
    lmain.after(10, show_frame)


def open_file():
    filepath = filedialog.askopenfilename()
    global cap

    if filepath != "":
        cap = cv2.VideoCapture(filepath)


def close():
    tk.destroy()
    tk.quit()


open_button = Button(imageFrame, text="Import file", command=open_file)
open_button.grid(row=100, column=0)
tk.protocol('WM_DELETE_WINDOW', close)

show_frame()
tk.mainloop()