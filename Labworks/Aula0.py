# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import imageForms
from tkinter import filedialog
import cv2 as cv
# open image


def open_show_image(pathname):
    img = cv.imread(pathname)
    cv.imshow("Imagem", img)
    return img


def file_chooser():
    #root = tk.Tk()
    #root.withdraw()
    file_path = filedialog.askopenfilename()
    return file_path


def grayscale(img):
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #cv.imshow("Imagem_Gray", img_gray)
    return img_gray


def bw(img_gray):
    ret,img_bw = cv.threshold(img_gray, 12, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    #cv.imshow("Imagem_BW", img_bw)
    return img_bw


def brightness_contrast(img):
    height = img.shape[0]
    width = img.shape[1]
    channels = img.shape[2]
    contrast = 5
    brightness = 5
    for x in range(0, width):
        for y in range(0, height):
            img[x, y] = img[x, y] * contrast + brightness
    return img

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #img = open_show_image()
    #img_gray = grayscale(img)
    #img_bw = bw(img_gray)
    #img_side_by_side = imageForms.showSideBySideImages(img, img, BGR1=True, BGR2=False )
    #ksize = (3, 3)
    #img_blur = cv.blur(img, ksize)
    #cv.imshow("Imagem_blur", img_blur)
    #cv.imshow("Imagem_brightness", brightness_contrast(img))

    file_path = file_chooser()
    img = open_show_image(file_path)
    cv.waitKey()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
