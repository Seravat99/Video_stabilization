import cv2 as cv
import ImageHough
import Aula0
import imageForms
import numpy as np


def showHoughPLane(img_original):

    img_grey = cv.cvtColor(img_original, cv.COLOR_BGR2GRAY)
    img_hough = ImageHough.HoughPlane(img_grey, 0, 360, 5)
    imageForms.showSideBySideImages(img_original, np.uint8(img_hough[0])*100, "Hough Plane", False, False)


def showHoughLinesMain(img, imgOriginal, rho, theta, thresh):
    img_hough_lines = ImageHough.ShowHoughLines(img, imgOriginal, rho, theta, thresh)
    imageForms.showSideBySideImages(img, img_hough_lines, "Hough Lines", False, False)


def ShowHoughLineSegmentsMain(img, imgOriginal, rho, theta, thresh, minLineLength, maxLineGap):
    # img = img gray
    imgCanny = cv.Canny(img, 100, 50)
    img_hough_line_segments = ImageHough.ShowHoughLineSegments(imgCanny, imgOriginal, rho, theta, thresh, minLineLength, maxLineGap)
    imageForms.showSideBySideImages(img_hough_line_segments, img, "Hough Lines Segments", False, False)


def ShowHoughCirclesMain(img, imgOriginal, param1):
    img_hough_circles = ImageHough.ShowHoughCircles(img, imgOriginal, param1)
    imageForms.showSideBySideImages(img_hough_circles, img, "Hough Circles", False, False)


def ex2():
    pathname = "..\\TAPDI_aulas\\images\\"
    filename = "aula4-3.bmp"
    img_original = Aula0.open_show_image(pathname+filename)
    img_gray = cv.cvtColor(img_original, cv.COLOR_BGR2GRAY)
    showHoughLinesMain(img_gray, img_original, 2, 0.0175, 5)


def ex3():
    # 3.1 3.2 3.3
    pathname = "..\\TAPDI_aulas\\images\\"
    filename = "aula4-2.bmp"
    img_original = Aula0.open_show_image(pathname + filename)
    img_gray = cv.cvtColor(img_original, cv.COLOR_BGR2GRAY)

    imgSobel = cv.Sobel(img_gray, cv.CV_8U, 1, 0)
    cv.threshold(imgSobel, 255, 255, cv.THRESH_OTSU + cv.THRESH_BINARY, imgSobel)
    showHoughLinesMain(imgSobel, img_original, 2, 0.0175, 120)


def ex4():
    # 4
    pathname = "..\\TAPDI_aulas\\images\\"
    filename = "aula4-2.bmp"
    img_original = Aula0.open_show_image(pathname + filename)
    img_gray = cv.cvtColor(img_original, cv.COLOR_BGR2GRAY)

    ShowHoughLineSegmentsMain(img_gray, img_original, 8, 0.0175, 120, 15, 1)


def ex5():
    # 5
    pathname = "..\\TAPDI_aulas\\images\\"
    filename = "aula4-coins.png"
    img_original = Aula0.open_show_image(pathname + filename)
    img_gray = cv.cvtColor(img_original, cv.COLOR_BGR2GRAY)

    ShowHoughCirclesMain(img_gray, img_original, 120)


def ex6():
    # 6
    filename = "aula4-video.mp4"
    ImageHough.ShowVideo(filename)


if __name__ == "__main__":
    exercise = int(input('What is the exercise?\n'))

    if exercise == 2:
        ex2()
    if exercise == 3:
        ex3()
    if exercise == 4:
        ex4()
    if exercise == 5:
        ex5()
    if exercise == 6:
        ex6()


