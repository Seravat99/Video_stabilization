import cv2 as cv
import Aula0 as a0
import imageForms as iF
import numpy as np
import time


def templateMatching(imgOriginal, template, method, mask=None):
    # imgOriginal – original image (WxH)
    # Temp - template image (twxth)
    # Result – image of type – size = W-tw+1xH-th+1
    # Method – calculation method (TM_SQDIFF, TM_SQDIFF_NORMED, TM_CCORR,
    # TM_CCORR_NORMED, TM_CCOEFF, TM_CCOEFF_NORMED)
    # Mask – apply only on certain pixels
    # Returns an image (single channel only) with the result of the Template Match.
    return cv.matchTemplate(imgOriginal, template, method, mask=None)


def minMaxLoc(imgOriginal):
    # Returns the value and location of the minimum and maximum values of the image for each channel.
    minVal, maxVal, minLoc, maxLoc = cv.minMaxLoc(imgOriginal)
    return minVal, maxVal, minLoc, maxLoc


def violaJones(imgOriginal, classifierPath, width, height):
    # Returns a list of rectangles that are listed as face. Allows you to set minimum and maximum size
    haar = cv.CascadeClassifier(classifierPath)
    faces = haar.detectMultiScale(imgOriginal, scaleFactor=1.4, minSize=(20, 20), maxSize=(width // 2, height // 2))
    return faces


def histogramOG(imgOriginal):
    # Histogram of Oriented Gradients (HOG)
    # Returns a list of detected objects (with rectangle), in this case of people.
    hog = cv.HOGDescriptor()
    hog.setSVMDetector(hog.getDefaultPeopleDetector())
    pedestrians = hog.detectMultiScale(imgOriginal)
    for (x, y, w, h) in pedestrians[0]:
        imgOriginal = cv.rectangle(imgOriginal, (x, y), (x + w, y + h), (0, 0, 255), 2)
    return imgOriginal


# Returns an image marked with rectangles and a list of detected faces.
def getFaceBox(frame, conf_threshold):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]

    # Open DNN model
    modelFile = "..\\TAPDI_aulas\\models\\opencv_face_detector_uint8.pb"
    configFile = "..\\TAPDI_aulas\\models\\opencv_face_detector.pbtxt"
    net = cv.dnn.readNetFromTensorflow(modelFile, configFile)
    # prepare for DNN
    blob = cv.dnn.blobFromImage(frameOpencvDnn, 1.0, (500, 300), [104, 117, 123], True, False)
    # set image as DNN input
    net.setInput(blob)
    # get Output
    detections = net.forward()
    bboxes = []
    x1 = 0
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            cv.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)
    return frameOpencvDnn, bboxes


# def detect(gray_s):
#     rects = face_detector.detectMultiScale(gray_s,
#                                            scaleFactor=1.1,
#                                            minNeighbors=5,
#                                            minSize=(30, 30),
#                                            flags=cv.CASCADE_SCALE_IMAGE)
#
#     for rect in rects:
#         cv.rectangle(gray_s, rect, 255, 2)
#
#     cap = cv.VideoCapture(0)
#     t0 = time.time()
#
#     M = np.float32([[0.5, 0, 0], [0, 0.5, 0]])
#     size = (640, 360)


def video_face_detection():
    cap = cv.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        frameOpencvDnn, bboxes = getFaceBox(frame)

        cv.imshow('window', frameOpencvDnn)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()


def video_pedestrian_detection():
    pathname = "..\\TAPDI_aulas\\images\\"
    filename = "pedestrian Video.mp4"
    video_path = pathname + filename

    vidCap = cv.VideoCapture(video_path)
    if not vidCap.isOpened():
        print("Video File Not Found")
        exit(-1)
    cv.namedWindow("Video")
    cv.startWindowThread()
    while cv.getWindowProperty("Video", cv.WND_PROP_VISIBLE) == 1:

        ret, vidFrame = vidCap.read()
        if not ret:
            break

        vidFrame = histogramOG(vidFrame)

        cv.imshow("Video", vidFrame)
        if cv.waitKey(20) >= 0:
            break
    cv.destroyAllWindows()


# Template matching
def ex1():
    imgOriginal = cv.imread("..\\TAPDI_aulas\\images\\road (4).bmp")#a0.open_show_image(a0.file_chooser())
    imgTemplate = cv.imread("..\\TAPDI_aulas\\images\\car template (2).bmp")#a0.open_show_image(a0.file_chooser())

    match_method = cv.TM_CCOEFF_NORMED
    template = templateMatching(imgOriginal, imgTemplate, method=match_method)
    cv.normalize(template, template, 0, 1, cv.NORM_MINMAX, -1)
    minVal, maxVal, minLoc, maxLoc = minMaxLoc(template)

    if match_method == cv.TM_SQDIFF or match_method == cv.TM_SQDIFF_NORMED:
        matchLoc = minLoc
    else:
        matchLoc = maxLoc
    imgCopy = imgOriginal.copy()
    cv.rectangle(imgCopy, pt1=matchLoc, pt2=(matchLoc[0] + imgTemplate.shape[0], matchLoc[1] + imgTemplate.shape[1]), color=(0, 0, 255), thickness=1)
    iF.showSideBySideImages(template, imgCopy, "Image draw", BGR1=False, BGR2=False)


# Viola Jones
def ex2():
    imgOriginal = a0.open_show_image(a0.file_chooser())
    imgCopy = imgOriginal.copy()
    classifierPath = "..\\TAPDI_aulas\\models\\haarcascade_frontalface_alt2.xml"
    width = 800
    height = 800
    faces = violaJones(imgOriginal, classifierPath, width, height)
    for rect in faces:
        cv.rectangle(imgCopy, rect, 255, 2)
    iF.showSideBySideImages(imgOriginal, imgCopy, "Face detection", BGR1=False, BGR2=False)


# Histogram of Oriented Gradients (HOG)
def ex3():
    imgOriginal = a0.open_show_image(a0.file_chooser())
    imgCopy = imgOriginal.copy()
    img_rect = histogramOG(imgCopy)
    iF.showSideBySideImages(imgOriginal, img_rect, "Face detection", BGR1=False, BGR2=False)


# DNN for face detector
def ex4():
    imgOriginal = a0.open_show_image(a0.file_chooser())
    frameOpencvDnn, bboxes = getFaceBox(imgOriginal, conf_threshold=0.4)
    iF.showSideBySideImages(imgOriginal, frameOpencvDnn, "Face detection", BGR1=False, BGR2=False)


if __name__ == "__main__":
    exercise = int(input('What is the exercise?\n'))

    if exercise == 1:
        ex1()
    if exercise == 2:
        ex2()
    if exercise == 3:
        ex3()
    if exercise == 4:
        ex4()
    if exercise == 5:
        video_face_detection()
    if exercise == 6:
        video_pedestrian_detection()


