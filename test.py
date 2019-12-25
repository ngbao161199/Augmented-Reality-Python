import sys
import cv2
import math
import dlib
import time
import numpy as np
import PySimpleGUI as sg

img_nose = cv2.imread("nose.png", -1)
img_nose_mask = img_nose[:, :, 3]
img_nose = img_nose[:, :, 0:3]

img_glasses = cv2.imread("glasses.png", -1)
img_glasses_mask = img_glasses[:, :, 3]
img_glasses = img_glasses[:, :, 0:3]

img_beyes = cv2.imread("demon_eyes.png", -1)
img_beyes_mask = img_beyes[:, :, 3]
img_beyes = img_beyes[:, :, 0:3]

img_moustache = cv2.imread("moustache.png", -1)
img_moustache_mask = img_moustache[:, :, 3]
img_moustache = img_moustache[:, :, 0:3]


def rotateImage(image, angle):
    h, w = image.shape[:2]
    img_c = (w / 2, h / 2)

    rot = cv2.getRotationMatrix2D(img_c, angle * 180 / math.pi, 1)

    sin = math.sin(angle)
    cos = math.cos(angle)
    b_w = int((h * abs(sin)) + (w * abs(cos)))
    b_h = int((h * abs(cos)) + (w * abs(sin)))

    rot[0, 2] += ((b_w / 2) - img_c[0])
    rot[1, 2] += ((b_h / 2) - img_c[1])

    outImg = cv2.warpAffine(image, rot, (b_w, b_h), flags=cv2.INTER_LINEAR)
    return outImg


def show(x1, y1, x2, y2, ow, oh, mask, img, frame):
    w = abs(x2 - x1)
    h = abs(y2 - y1)

    if h > 0 and w > 0:
        ow = w
        oh = h

    mask = cv2.resize(mask, (ow, oh))
    img = cv2.resize(img, (ow, oh))
    mask_inv = cv2.bitwise_not(mask)
    roi = frame[y1:y2, x1:x2]

    roi_bakground = cv2.bitwise_and(roi, roi, mask=mask_inv)
    roi_foreground = cv2.bitwise_and(img, img, mask=mask)
    frame[y1:y2, x1:x2] = cv2.add(roi_bakground, roi_foreground)


def test_pos(landmarks, frame, n):
    x = landmarks[n].x
    y = landmarks[n].y
    cv2.circle(frame, (x, y), 4, (255, 0, 0), -1)


def main():
    sg.change_look_and_feel('LightGreen')

    layout = [
        [sg.Text('OpenCV Demo - SSnap', size=(40, 1), justification='center')],
        [sg.Image(filename='', key='image')],
        [sg.Button('Pig nose', size=(15, 1), key='nose'),
         sg.Button('Glasses', size=(15, 1), key='glass'),
         sg.Button('Mắt biếc', size=(15, 1), key='beyes'),
         sg.Button('Moustache', size=(15, 1), key='rau')],
        [sg.Button('Show landmarks', size=(15, 1), key='show landmarks')],
        [sg.Button('Exit', size=(15, 1))]
    ]

    window = sg.Window('OpenCV Demo - SSnap',
                       layout,
                       location=(800, 600),
                       finalize=True)

    cap = cv2.VideoCapture(0)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    #cap = cv2.VideoCapture(0)

    flag_nose = False
    flag_glass = False
    flag_rau = False
    flag_landmarks = False
    flag_beyes = False
    dpoint = dlib.dpoint(0, 0)
    old_landmarks = 0
    old_landmarks1 = 0
    flag_landmarks1 = True

    while True:
        start_time = time.time()

        event, _ = window.read(timeout=0, timeout_key='timeout')

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
                landmarks = np.array(landmarks.parts())
                ddpoint = landmarks[30]
                if math.hypot(ddpoint.x - dpoint.x, ddpoint.y - dpoint.y) > 10:
                    old_landmarks = landmarks
                    dpoint = ddpoint
                    flag_landmarks1 = True
                else:
                    landmarks = old_landmarks
                    # print("sau khi doi", landmarks[30],old_landmarks[30])
                    flag_landmarks1 = False

                min_x = min_y = sys.maxsize
                max_x = max_y = -sys.maxsize
                for i in range(27):
                    a = landmarks[i]
                    min_x = min(a.x, min_x)
                    min_y = min(a.y, min_y)
                    max_x = max(a.x, max_x)
                    max_y = max(a.y, max_y)
                # cv2.circle(frame, (min_x, min_y), 4, (0, 255, 0), -1)
                # cv2.circle(frame, (max_x, max_y), 4, (0, 255, 0), -1)

                min_x = max(min_x - 50, 0)
                min_y = max(min_y - 50, 0)
                max_x += 50
                max_y += 50
                new_face = frame[min_y:max_y, min_x:max_x]
                nw, nh = new_face.shape[:2]
                dpo = landmarks[26] - landmarks[17]
                angle = math.atan2(0, 1) - math.atan2(dpo.y, dpo.x)
                new_face = rotateImage(new_face, -angle)
                gray1 = cv2.cvtColor(new_face, cv2.COLOR_BGR2GRAY)
                fac = detector(gray1)
                for fa in fac:
                    landmarks = predictor(gray1, fa)
                    landmarks = np.array(landmarks.parts())
                    if flag_landmarks1:
                        old_landmarks1 = landmarks
                    else:
                        landmarks = old_landmarks1
                        # print("sau khi doi",landmarks[30],old_landmarks1[30])
                    if flag_nose:
                        show(
                            landmarks[31].x,
                            landmarks[29].y,
                            landmarks[35].x,
                            landmarks[33].y, ow, oh, img_nose_mask,
                            img_nose, new_face)
                    if flag_glass:
                        show(
                            landmarks[17].x,
                            landmarks[17].y,
                            landmarks[26].x,
                            landmarks[30].y, ow, oh, img_glasses_mask,
                            img_glasses, new_face)
                    if flag_beyes:
                        show(
                            landmarks[17].x,
                            landmarks[17].y,
                            landmarks[26].x,
                            landmarks[30].y, ow, oh, img_beyes_mask,
                            img_beyes, new_face)
                    if flag_rau:
                        show(
                            landmarks[48].x,
                            landmarks[33].y,
                            landmarks[54].x,
                            landmarks[52].y, ow, oh, img_moustache_mask,
                            img_moustache, new_face)
                    if flag_landmarks:
                        for n in range(0, 68):
                            test_pos(landmarks, new_face, n)

                new_face = rotateImage(new_face, angle)
                dnw, dnh = new_face.shape[:2]
                dnw = int((dnw - nw) / 2)
                dnh = int((dnh - nh) / 2)
                new_face = new_face[dnw + 5:dnw + nw - 5, dnh + 5:dnh + nh - 5]
                dnw, dnh = new_face.shape[:2]
                frame[min_y + 5:min_y + min(dnw, nw - 5) + 5,
                      min_x + 5:min_x + min(dnh, nh - 5) + 5] = new_face

        cv2.putText(frame, str(round(1.0/(time.time() - start_time),3)), (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
        imgbytes = cv2.imencode('.png', frame)[1].tobytes()
        window['image'].update(data=imgbytes)

        #print("FPS: ", 1.0 / (time.time() - start_time))
        

    window.close()


main()
