import numpy as np
import cv2 as cv

import sys


BLUE = [255, 0, 0]  # rectangle color
RED = [0, 0, 255]  # PR BG
GREEN = [0, 255, 0]  # PR FG
BLACK = [0, 0, 0]  # sure BG
WHITE = [255, 255, 255]  # sure FG

DRAW_BG = {"color": BLACK, "val": 0}
DRAW_FG = {"color": WHITE, "val": 1}
DRAW_PR_BG = {"color": RED, "val": 2}
DRAW_PR_FG = {"color": GREEN, "val": 3}

# setting up flags
rect = (0, 0, 1, 1)
drawing = False  # flag for drawing curves
rectangle = False  # flag for drawing rect
rect_over = False  # flag to check if rect drawn
rect_or_mask = 100  # flag for selecting rect or mask mode
value = DRAW_FG  # drawing initialized to FG
thickness = 3  # brush thickness


def onmouse(event, x, y, flags, param):

    global img
    global img2
    global drawing
    global value
    global mask
    global rectangle
    global rect
    global rect_or_mask
    global ix, iy
    global rect_over
    global flag
    # Draw Rectangle
    if event == cv.EVENT_RBUTTONDOWN:
        rectangle = True
        ix, iy = x, y

    elif event == cv.EVENT_MOUSEMOVE:
        if rectangle == True:
            img = img2.copy()
            cv.rectangle(img, (ix, iy), (x, y), BLUE, 2)
            rect = (
                min(ix, x),
                min(iy, y),
                abs(ix - x),
                abs(iy - y),
            )
            rect_or_mask = 0

    elif event == cv.EVENT_RBUTTONUP:
        rectangle = False
        rect_over = True
        cv.rectangle(img, (ix, iy), (x, y), BLUE, 2)
        rect = (
            min(ix, x),
            min(iy, y),
            abs(ix - x),
            abs(iy - y),
        )
        rect_or_mask = 0
        print(" Now press the key 'n' a few times until no further change \n")

    # draw touchup curves

    if event == cv.EVENT_LBUTTONDOWN:
        if rect_over == False:
            print("first draw rectangle \n")
        else:
            drawing = True
            cv.circle(img, (x, y), thickness, value["color"], -1)
            cv.circle(mask, (x, y), thickness, value["val"], -1)

    elif event == cv.EVENT_MOUSEMOVE:
        if drawing == True:
            cv.circle(img, (x, y), thickness, value["color"], -1)
            cv.circle(mask, (x, y), thickness, value["val"], -1)

    elif event == cv.EVENT_LBUTTONUP:
        if drawing == True:
            drawing = False
            cv.circle(img, (x, y), thickness, value["color"], -1)
            cv.circle(mask, (x, y), thickness, value["val"], -1)

    else:
        drawing = False
        rectangle = False


# Loading images
if len(sys.argv) == 2:
    filename = sys.argv[1]  # for drawing purposes
else:
    print("No input image given, so loading default image, lena.jpg \n")
    print("Correct Usage: python grabcut.py <filename> \n")
    filename = "default.jpg"


# Load the Image
img = cv.imread(cv.samples.findFile(filename))

# resize the image
img = cv.resize(img, (960, 960))
# a copy of original image
img2 = img.copy()
# mask initialized to PR_BG
mask = np.zeros(img.shape[:2], dtype=np.uint8)
# output image to be shown
output = np.zeros(img.shape, np.uint8)


print(" Instructions: \n")
print(" Draw a rectangle around the object using right mouse button \n")
# input and output windows
cv.namedWindow("output")
cv.namedWindow("input")
cv.setMouseCallback("input", onmouse)
cv.moveWindow("input", img.shape[1] + 10, 90)


while 1:

    k = cv.waitKey(1)

    # key bindings
    if k == 27:  # esc to exit
        break
    elif k == ord("0"):  # BG drawing
        print(" mark background regions with left mouse button \n")
        value = DRAW_BG
    elif k == ord("1"):  # FG drawing
        print(" mark foreground regions with left mouse button \n")
        value = DRAW_FG
    elif k == ord("2"):  # PR_BG drawing
        value = DRAW_PR_BG
    elif k == ord("3"):  # PR_FG drawing
        value = DRAW_PR_FG
    elif k == ord("s"):  # save image
        bar = np.zeros((img.shape[0], 5, 3), np.uint8)
        res = np.hstack((img2, bar, img, bar, output))
        cv.imwrite("grabcut_output.png", res)
        print(" Result saved as image \n")
    elif k == ord("r"):  # reset everything
        print("resetting \n")
        rect = (0, 0, 1, 1)
        drawing = False
        rectangle = False
        rect_or_mask = 100
        rect_over = False
        value = DRAW_FG
        img = img2.copy()
        # mask initialized to PR_BG
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        # output image to be shown
        output = np.zeros(img.shape, np.uint8)
    elif k == ord("n"):  # segment the image
        print(
            """ For finer touchups, mark foreground and background after pressing keys 0-3
                and again press 'n' \n"""
        )
        try:
            bgdmodel = np.zeros((1, 65), np.float64)
            fgdmodel = np.zeros((1, 65), np.float64)
            if rect_or_mask == 0:  # grabcut with rect
                cv.grabCut(
                    img2, mask, rect, bgdmodel, fgdmodel, 1, cv.GC_INIT_WITH_RECT
                )
                rect_or_mask = 1
            elif rect_or_mask == 1:  # grabcut with mask
                cv.grabCut(
                    img2, mask, rect, bgdmodel, fgdmodel, 1, cv.GC_INIT_WITH_MASK
                )
        except:
            import traceback

            traceback.print_exc()

    mask2 = np.where((mask == 1) + (mask == 3), 255, 0).astype("uint8")
    output = cv.bitwise_and(img2, img2, mask=mask2)

    background = img - output
    background[np.where((background > [0, 0, 0]).all(axis=2))] = [255, 255, 255]
    white_bg = background + output

    # try:
    #     bg_img = cv.imread("yellow_bg.png")
    #     img2 = white_bg
    #     bg_img = cv.resize(bg_img, (960, 960))
    # except Exception as e:
    #     print(str(e))

    # rows, cols, channels = img2.shape
    # roi = bg_img[0:rows, 0:cols]
    # img2gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    # ret, mask = cv.threshold(img2gray, 220, 255, cv.THRESH_BINARY_INV)
    # mask_inv = cv.bitwise_not(mask)
    # img1_bg = cv.bitwise_and(roi, roi, mask=mask_inv)
    # img2_fg = cv.bitwise_and(img2, img2, mask=mask)
    # dst = cv.add(img1_bg, img2_fg)
    # bg_img[0:rows, 0:cols] = dst

    # bg = np.zeros_like(img)
    # bg[:, :, :] = (12, 198, 150)
    # bg = cv.bitwise_and(background, cv.bitwise_not(mask2))

    # cv.imshow("new Back", bg_img)
    cv.imshow("output", output)
    # cv.imshow('white_back', white_bg)
    cv.imshow("input", img)
    cv.imshow("back", white_bg)

    k = cv.waitKey(1)
{"mode": "full", "isActive": false}
