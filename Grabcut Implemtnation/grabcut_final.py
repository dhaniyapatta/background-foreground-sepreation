#!/usr/bin/env python
"""
===============================================================================
Interactive Image Segmentation using GrabCut algorithm.

This sample shows interactive image segmentation using grabcut algorithm.

USAGE:
    python grabcut.py <filename>


Key '0' - To select areas of sure background
Key '1' - To select areas of sure foreground
Key '2' - To select areas of probable background
Key '3' - To select areas of probable foreground

Key 'n' - To update the segmentation
Key 'r' - To reset the setup
Key 's' - To save the results
===============================================================================
"""

# Python 2/3 compatibility
from __future__ import print_function
from keras.preprocessing import image

import numpy as np
import cv2 as cv

import sys


class App:
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
    thickness = 6  # brush thickness

    def onmouse(self, event, x, y, flags, param):
        # Draw Rectangle
        if event == cv.EVENT_RBUTTONDOWN:
            self.rectangle = True
            self.ix, self.iy = x, y

        elif event == cv.EVENT_MOUSEMOVE:
            if self.rectangle == True:
                self.img = self.img2.copy()
                cv.rectangle(self.img, (self.ix, self.iy), (x, y), self.BLUE, 2)
                self.rect = (
                    min(self.ix, x),
                    min(self.iy, y),
                    abs(self.ix - x),
                    abs(self.iy - y),
                )
                self.rect_or_mask = 0

        elif event == cv.EVENT_RBUTTONUP:
            self.rectangle = False
            self.rect_over = True
            cv.rectangle(self.img, (self.ix, self.iy), (x, y), self.BLUE, 2)
            self.rect = (
                min(self.ix, x),
                min(self.iy, y),
                abs(self.ix - x),
                abs(self.iy - y),
            )
            self.rect_or_mask = 0
            print(" Now press the key 'n' a few times until no further change \n")

        # draw touchup curves

        if event == cv.EVENT_LBUTTONDOWN:
            if self.rect_over == False:
                print("first draw rectangle \n")
            else:
                self.drawing = True
                cv.circle(self.img, (x, y), self.thickness, self.value["color"], -1)
                cv.circle(self.mask, (x, y), self.thickness, self.value["val"], -1)

        elif event == cv.EVENT_MOUSEMOVE:
            if self.drawing == True:
                cv.circle(self.img, (x, y), self.thickness, self.value["color"], -1)
                cv.circle(self.mask, (x, y), self.thickness, self.value["val"], -1)

        elif event == cv.EVENT_LBUTTONUP:
            if self.drawing == True:
                self.drawing = False
                cv.circle(self.img, (x, y), self.thickness, self.value["color"], -1)
                cv.circle(self.mask, (x, y), self.thickness, self.value["val"], -1)

    def run(self):
        # Loading images
        if len(sys.argv) == 2:
            filename = sys.argv[1]  # for drawing purposes
        else:
            print("No input image given, so loading default image, lena.jpg \n")
            print("Correct Usage: python grabcut.py <filename> \n")
            filename = "lena.jpg"

        self.img = cv.imread(cv.samples.findFile(filename))

        # resize image
        self.img = cv.resize(self.img, (960, 960))
        # a copy of original image
        self.img2 = self.img.copy()
        # mask initialized to PR_BG
        self.mask = np.zeros(self.img.shape[:2], dtype=np.uint8)
        # output image to be shown
        self.output = np.zeros(self.img.shape, np.uint8)

        # input and output windows
        cv.namedWindow("output")
        cv.namedWindow("input")
        cv.setMouseCallback("input", self.onmouse)
        cv.moveWindow("input", self.img.shape[1] + 10, 90)

        print(" Instructions: \n")
        print(" Draw a rectangle around the object using right mouse button \n")

        while 1:

            cv.imshow("output", self.output)

            # cv.imshow('white_back', self.white_bg)
            cv.imshow("input", self.img)

            k = cv.waitKey(1)

            # key bindings
            if k == 27:  # esc to exit
                break
            elif k == ord("0"):  # BG drawing
                print(" mark background regions with left mouse button \n")
                self.value = self.DRAW_BG
            elif k == ord("1"):  # FG drawing
                print(" mark foreground regions with left mouse button \n")
                self.value = self.DRAW_FG
            elif k == ord("2"):  # PR_BG drawing
                self.value = self.DRAW_PR_BG
            elif k == ord("3"):  # PR_FG drawing
                self.value = self.DRAW_PR_FG
            elif k == ord("s"):  # save image
                bar = np.zeros((self.img.shape[0], 5, 3), np.uint8)
                res = np.hstack(
                    (self.img2, bar, self.img, bar, self.output, self.white_bg)
                )
                cv.imwrite("grabcut_output.png", res)
                print(" Result saved as image \n")
            elif k == ord("r"):  # reset everything
                print("resetting \n")
                self.rect = (0, 0, 1, 1)
                self.drawing = False
                self.rectangle = False
                self.rect_or_mask = 100
                self.rect_over = False
                self.value = self.DRAW_FG
                self.img = self.img2.copy()
                # mask initialized to PR_BG
                self.mask = np.zeros(self.img.shape[:2], dtype=np.uint8)
                # output image to be shown
                self.output = np.zeros(self.img.shape, np.uint8)
            elif k == ord("n"):  # segment the image
                print(
                    """ For finer touchups, mark foreground and background after pressing keys 0-3
                and again press 'n' \n"""
                )
                try:
                    bgdmodel = np.zeros((1, 65), np.float64)
                    fgdmodel = np.zeros((1, 65), np.float64)
                    if self.rect_or_mask == 0:  # grabcut with rect
                        cv.grabCut(
                            self.img2,
                            self.mask,
                            self.rect,
                            bgdmodel,
                            fgdmodel,
                            1,
                            cv.GC_INIT_WITH_RECT,
                        )
                        self.rect_or_mask = 1
                    elif self.rect_or_mask == 1:  # grabcut with mask
                        cv.grabCut(
                            self.img2,
                            self.mask,
                            self.rect,
                            bgdmodel,
                            fgdmodel,
                            1,
                            cv.GC_INIT_WITH_MASK,
                        )
                except:
                    import traceback

                    traceback.print_exc()

            mask2 = np.where((self.mask == 1) + (self.mask == 3), 255, 0).astype(
                "uint8"
            )
            self.output = cv.bitwise_and(self.img2, self.img2, mask=mask2)
            self.background = self.img - self.output
            self.background[np.where((self.background > [0, 0, 0]).all(axis=2))] = [
                255,
                255,
                255,
            ]
            self.white_bg = self.background + self.output

            self.bg_img = cv.imread("yellow_bg.png")
            img2 = self.white_bg
            self.bg_img = cv.resize(self.bg_img, (960, 960))
            rows, cols, channels = img2.shape
            roi = self.bg_img[0:rows, 0:cols]
            img2gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

            ret, mask = cv.threshold(img2gray, 220, 255, cv.THRESH_BINARY_INV)
            mask_inv = cv.bitwise_not(mask)
            img1_bg = cv.bitwise_and(roi, roi, mask=mask_inv)
            img2_fg = cv.bitwise_and(img2, img2, mask=mask)
            dst = cv.add(img1_bg, img2_fg)
            self.bg_img[0:rows, 0:cols] = dst
            cv.imshow("new Back", self.white_bg)
        print("Done")


if __name__ == "__main__":
    print(__doc__)
    App().run()
    cv.destroyAllWindows()
{"mode": "full", "isActive": false}
