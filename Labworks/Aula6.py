import cv2 as cv
import numpy as np
import ImageDeconvolution as id
import imageForms as iF
import Aula0 as a0
import imgeOpticalFlow as iOF


def blurImage(imgGray, motion, cutOff):
    filter = id.GetFilterConv(motion, cutOff)
    imgResGauss = cv.filter2D(imgGray, cv.CV_8U, filter, anchor=(np.int32(filter.shape[0] / 2), np.int32(filter.shape[1] / 2)))
    return imgResGauss


def whiteNoiseImage(imgOriginal):
    noiseAmp = int(input('What is the noise intensity?\n'))
    imgRand = np.random.rand(imgOriginal.shape[0], imgOriginal.shape[1]) * noiseAmp
    return imgRand


def inverseDeconvolutionButterworth(imgRes, motion, cutOff):
    return id.InverseDeconvolutionButterworth(imgRes, motion, cutOff, cutOffBW=50)


def inverseDeconvolutionWiener(imgInput, kFactor, motion, cutOff):
    return id.InverseDeconvolutionWiener(imgInput, kFactor, motion, cutOff)


def ex1():
    # 2. Blur the image with a Gaussian averaging filter and with a Motion Blur filter (GetFilterConv and cv.filter2D).
    pathname = "..\\TAPDI_aulas\\images\\"
    filename = "Aula 6 (1).png"
    imgOriginal = cv.imread(pathname + filename)
    if imgOriginal is None:
        print("Image File Not Found")
        exit(-1)
    imgGray = a0.grayscale(imgOriginal)
    cutOff = int(input('What is the cutoff?\n'))
    motion = False
    imgResGauss = blurImage(imgGray, motion, cutOff)
    iF.showSideBySideImages(imgOriginal, imgResGauss, "Gauss Blur", BGR1=False, BGR2=False)


def ex2():
    # 3. Add white noise to the image. Ask the user the intensity of noise (e.g. 10).
    pathname = "..\\TAPDI_aulas\\images\\"
    filename = "Aula 6 (1).png"
    imgOriginal = cv.imread(pathname + filename)
    imgWhiteNoise = whiteNoiseImage(imgOriginal)
    iF.showSideBySideImages(imgOriginal, imgWhiteNoise, "White Noise", BGR1=False, BGR2=False)


def ex3():
    # 4. Apply the image debluring method by inverse Deconvolution with Butterworth Filtering to an
    # image that is blurred by a Gaussian average filter (degradation function).
    # (InverseDeconvolutionButterworth)
    pathname = "..\\TAPDI_aulas\\images\\"
    filename = "Aula 6 (1).png"
    imgOriginal = cv.imread(pathname + filename)
    if imgOriginal is None:
        print("Image File Not Found")
        exit(-1)
    motion = False
    imgGray = a0.grayscale(imgOriginal)
    cutOff = float(input('What is the cutoff?\n'))
    imgResGauss = blurImage(imgGray, motion, cutOff)
    imgNoise = whiteNoiseImage(imgResGauss)
    imgRes = imgResGauss + imgNoise

    imgDeconv = inverseDeconvolutionButterworth(imgRes, motion, cutOff)
    iF.showSideBySideImages(imgResGauss, imgDeconv, "Deconvolution with Butterworth", BGR1=False, BGR2=False)


def ex4():
    pathname = "..\\TAPDI_aulas\\images\\"
    filename = "Aula 6 (2).png"
    imgOriginal = cv.imread(pathname + filename)
    if imgOriginal is None:
        print("Image File Not Found")
        exit(-1)
    motion = False
    imgGray = a0.grayscale(imgOriginal)
    cutOff = int(input('What is the cutoff?\n'))
    kFactor = float(input('What is the kFactor?\n'))
    imgResGauss = blurImage(imgGray, motion, cutOff)
    imgNoise = whiteNoiseImage(imgGray)
    imgRes = imgResGauss + imgNoise
    imgDeconvWeiner = inverseDeconvolutionWiener(imgRes, kFactor, motion, cutOff)
    iF.showSideBySideImages(imgOriginal, imgDeconvWeiner, "Deconvolution with Butterworth", BGR1=False, BGR2=False)


def ex5():
    pathname = "..\\TAPDI_aulas\\images\\"
    filename = "Aula 6 (2).png"
    imgOriginal = cv.imread(pathname + filename)
    if imgOriginal is None:
        print("Image File Not Found")
        exit(-1)
    motion = True
    imgGray = a0.grayscale(imgOriginal)
    cutOff = int(input('What is the cutoff?\n'))
    kFactor = float(input('What is the kFactor?\n'))
    imgResGauss = blurImage(imgGray, motion, cutOff)
    imgNoise = whiteNoiseImage(imgGray)
    imgRes = imgResGauss + imgNoise
    imgDeconvWeiner = inverseDeconvolutionWiener(imgRes, kFactor, motion, cutOff)
    iF.showSideBySideImages(imgResGauss, imgDeconvWeiner, "Deconvolution with Butterworth", BGR1=False, BGR2=False)


def ex6():
    pathname = "..\\TAPDI_aulas\\images\\"
    filename = "aula4-video.mp4"
    video_path = pathname+filename
    iOF.LucasKanade_OF(video_path, 6)
    #iOF.Farneback_OF(video_path)


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
        ex5()
    if exercise == 6:
        ex6()
