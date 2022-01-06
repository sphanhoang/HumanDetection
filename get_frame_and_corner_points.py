import numpy as np
import cv2 as cv


def get_frame():
    cap = cv.VideoCapture(0)
    print("[get frame] Camera is on...")
    while True:
        isTrue, frame = cap.read()

        resized = cv.resize(frame, (int(frame.shape[1] * 1), int(frame.shape[0] * 1)),
                            interpolation=cv.INTER_CUBIC)


        cv.imshow('Video', resized)
        if cv.waitKey(16) & 0xFF == ord('q'):
            cv.imwrite('frame.jpg', resized)
            print("[MAIN] Exiting program...")
            break
    cap.release()
    cv.destroyAllWindows()


def get_points(img):

    print("Double click to select point, press c to print, press q to quit")
    def draw_circle(event, x, y, flags, param):
        global mouseX, mouseY
        if event == cv.EVENT_LBUTTONDBLCLK:
            cv.circle(img, (x, y), 5, (0, 0, 255), -1)
            mouseX, mouseY = x, y

    img = cv.imread(img)
    cv.namedWindow('image')
    print("[get points] Select points...")
    cv.setMouseCallback('image', draw_circle)
    point = 1
    while (1):
        cv.imshow('image', img)
        k = cv.waitKey(20) & 0xFF
        if k == ord("q"):
            break
        elif k == ord('c'):
            print("p{}: ({}, {})".format(point, mouseX, mouseY))
            point = point+1

def menu():
    user_input = input("Select function: \n"
          "1. get_frame \n"
          "2. get points \n"
            "Waiting for selection... ")
    if user_input == "1":
        get_frame()
    elif user_input == "2":
        static_img = "./frame.jpg"
        get_points(static_img)
    else:
        print("Choose 1 or 2 only \n")
        menu()


menu()
