import PySimpleGUI as sg
import cv2
import numpy as np
import dlib
import math

# pig nose loaded
img_nose = cv2.imread("nose.png", -1)
img_nose_mask = img_nose[:, :, 3]
img_nose = img_nose[:, :, 0:3]

# glass loaded
img_glasses = cv2.imread("glasses.png", -1)
img_glasses_mask = img_glasses[:, :, 3]
img_glasses = img_glasses[:, :, 0:3]

# blue eyes image loaded
img_beyes = cv2.imread("demon_eyes.png", -1)
img_beyes_mask = img_beyes[:, :, 3]
img_beyes = img_beyes[:, :, 0:3]

# moustache loaded
img_moustache = cv2.imread("moustache.png", -1)
img_moustache_mask = img_moustache[:, :, 3]
img_moustache = img_moustache[:, :, 0:3]


def show(x1, y1, x2, y2, ow, oh, mask, img, frame):
    mask_inv = cv2.bitwise_not(mask)
    roi = frame[y1:y2, x1:x2]

    roi_bakground = cv2.bitwise_and(roi, roi, mask=mask_inv)
    roi_foreground = cv2.bitwise_and(img, img, mask=mask)
    frame[y1:y2, x1:x2] = cv2.add(roi_bakground, roi_foreground)

def show_beyes(landmarks, frame, ow, oh):
    x1 = landmarks.part(17).x
    y1 = landmarks.part(17).y
    x2 = landmarks.part(26).x
    y2 = landmarks.part(30).y
    w = abs(x2 - x1)
    h = abs(y2 - y1)

    if h > 0 and w > 0:
        ow = w
        oh = h

    mask = cv2.resize(img_beyes_mask, (ow, oh))
    img = cv2.resize(img_beyes, (ow, oh))

    # cv2.imshow("mask", mask)
    # cv2.imshow("img", img)

    show(x1, y1, x2, y2, ow, oh, mask, img, frame)

def show_nose(landmarks, frame, ow, oh):
    x1 = landmarks.part(31).x
    y1 = landmarks.part(29).y
    x2 = landmarks.part(35).x
    y2 = landmarks.part(33).y
    w = abs(x2 - x1)
    h = abs(y2 - y1)

    if h > 0 and w > 0:
        ow = w
        oh = h

    mask = cv2.resize(img_nose_mask, (ow, oh))
    img = cv2.resize(img_nose, (ow, oh))

    # cv2.imshow("mask", mask)
    # cv2.imshow("img", img)

    show(x1, y1, x2, y2, ow, oh, mask, img, frame)

def show_glasses(landmarks, frame, ow, oh):
    x1 = landmarks.part(17).x
    y1 = landmarks.part(17).y
    x2 = landmarks.part(26).x
    y2 = landmarks.part(30).y
    w = abs(x2 - x1)
    h = abs(y2 - y1)

    if h > 0 and w > 0:
        ow = w
        oh = h

    mask = cv2.resize(img_glasses_mask, (ow, oh))
    img = cv2.resize(img_glasses, (ow, oh))

    # cv2.imshow("mask", mask)
    # cv2.imshow("img", img)

    show(x1, y1, x2, y2, ow, oh, mask, img, frame)

def show_moustache(landmarks, frame, ow, oh):
    x1 = landmarks.part(48).x
    y1 = landmarks.part(33).y
    x2 = landmarks.part(54).x
    y2 = landmarks.part(52).y
    w = abs(x2 - x1)
    h = abs(y2 - y1)

    if h > 0 and w > 0:
        ow = w
        oh = h

    mask = cv2.resize(img_moustache_mask, (ow, oh))
    img = cv2.resize(img_moustache, (ow, oh))

    # cv2.imshow("mask", mask)
    # cv2.imshow("img", img)

    show(x1, y1, x2, y2, ow, oh, mask, img, frame)


def test_pos(landmarks, frame, n):
    x = landmarks.part(n).x
    y = landmarks.part(n).y
    cv2.circle(frame, (x, y), 4, (255, 0, 0), -1)


def main():
    sg.change_look_and_feel('LightGreen')

    # define the window layout
    layout = [
        [sg.Text('OpenCV Demo - SSnap', size=(40, 1), justification='center')],
        [sg.Image(filename='', key='image')],
        [sg.Button('Pig nose', size=(15, 1), key='nose')],
        [sg.Button('Glasses', size=(15, 1), key='glass')],
        [sg.Button('Stupid eyes', size=(15, 1), key='beyes')],
        [sg.Button('Moustache', size=(15, 1), key='rau')],
        [sg.Button('Show landmarks', size=(15, 1), key='show landmarks')],
        [sg.Button('Exit', size=(15, 1))]
    ]

    # create the window and show it without the plot
    window = sg.Window('OpenCV Demo - SSnap',
                       layout,
                       location=(800, 400),
                       finalize=True)

    cap = cv2.VideoCapture(0)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    cap = cv2.VideoCapture(0)
    
    flag_nose = True
    flag_glass = True
    flag_rau = True
    flag_landmarks = True
    flag_beyes = True

    while True:
        # window read will give us 2 value, so we need to declare 2 variables
        event, value = window.read(timeout=0, timeout_key='timeout')

        # condition to flag each event when we click
        if event == 'Exit' or event is None:
            break
        if event == 'nose':
            flag_nose = not flag_nose
        if event == 'glass':
            flag_glass = not flag_glass
        if event == 'beyes':
            flag_beyes = not flag_beyes
        if event == 'rau':
            flag_rau = not flag_rau
        if event == 'show landmarks':
            flag_landmarks = not flag_landmarks
        
        ret, frame = cap.read()

        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = detector(gray)
            ow = 0
            oh = 0
            for face in faces:
                landmarks = predictor(gray, face)

                if flag_nose:
                    show_nose(landmarks, frame, ow, oh)
                if flag_glass:
                    show_glasses(landmarks, frame, ow, oh)
                if flag_beyes:
                    show_beyes(landmarks, frame, ow, oh)
                if flag_rau:
                    show_moustache(landmarks, frame, ow, oh)
                if flag_landmarks:
                    for n in range(0, 68):
                        test_pos(landmarks, frame, n)

        imgbytes = cv2.imencode('.png', frame)[1].tobytes()
        window['image'].update(data=imgbytes)

    window.close()


main()