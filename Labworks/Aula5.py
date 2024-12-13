import Aula0 as a0
import ImageFFT as fft
import cv2 as cv
import numpy as np
import imageForms


def imageFFTSpectrum(img_gray):
    img_fft = fft.GetFFT_Mag_Phase(img_gray)
    img_log = [cv.log(img_fft[0]), img_fft[1]]
    return np.fft.fftshift(img_log)


def runFFT(img_gray):
    img_fft = imageFFTSpectrum(img_gray)
    imageForms.showSideBySideImages(img_fft[1], img_fft[0], 'fft')


def imageFFT(img_gray):
    return fft.GetFFT_Mag_Phase(img_gray)


def imageFFTInverse(mag, phase):
    return fft.GetFFT_Inverse_Mag_Phase(mag, phase)


def imgFFTExchange(img_gray_1, img_gray_2):
    img_fft_1 = imageFFT(img_gray_1)
    img_fft_2 = imageFFT(img_gray_2)

    img_fft_inv_m1 = imageFFTInverse(img_fft_2[0], img_fft_1[1])
    img_fft_inv_m2 = imageFFTInverse(img_fft_1[0], img_fft_2[1])
    imageForms.showSideBySideImages(img_fft_inv_m1, img_fft_inv_m2)


def idealFilter(img_gray_1):
    threshold = int(input('What is the threshold?\n'))
    img_fft = imageFFT(img_gray_1)
    mask = fft.CreateFilterMask_Ideal(img_fft[0].shape, threshold, False)
    img_masked = cv.multiply(img_fft[0], np.fft.fftshift(mask))
    return imageFFTInverse(img_masked, img_fft[1])


def gaussianFilter(img_gray_1):
    threshold = int(input('What is the threshold?\n'))
    img_fft = imageFFT(img_gray_1)
    mask = fft.CreateFilterMask_Gaussian(img_fft[0].shape, threshold, False)
    img_masked = cv.multiply(img_fft[0], np.fft.fftshift(mask))
    return imageFFTInverse(img_masked, img_fft[1])


def focusingApp():
    prev_set_1_mag = 0
    prev_set_2_mag = 0

    set_1_best = 0
    set_2_best_ = 0
    for k in range(1, 10):
        if k <= 8:
            img_original_set_1 = a0.open_show_image("..\\TAPDI_aulas\\images\\focus (" + str(k) + ").jpg")
            img_gray_set_1 = a0.grayscale(img_original_set_1)
            img_gray_set_1_mag = fft.GetFFT_Mag_Phase(img_gray_set_1)[0]
            set_1_mag = np.mean(img_gray_set_1_mag[round(img_gray_set_1_mag.__len__() / 2)])
            if set_1_mag > prev_set_1_mag:
                prev_set_1_mag = set_1_mag
                set_1_best = k
        img_original_set_2 = a0.open_show_image("..\\TAPDI_aulas\\images\\focus2 (" + str(k) + ").jpg")
        img_gray_set_2 = a0.grayscale(img_original_set_2)
        img_gray_set_2_mag = fft.GetFFT_Mag_Phase(img_gray_set_2)[0]
        set_2_mag = np.mean(img_gray_set_2_mag[round(img_gray_set_2_mag.__len__() / 2)])
        if set_2_mag > prev_set_2_mag:
            prev_set_2_mag = set_2_mag
            set_2_best = k

    print('set1 most focused is ' + str(set_1_best))
    print('set2 most focused is ' + str(set_2_best))


# shape mag vs filer mask
if __name__ == "__main__":
    #img_original_m_1 = a0.open_show_image(a0.file_chooser())
    #img_gray_m_1 = a0.grayscale(img_original_m_1)
    # runFFT(img_gray_m_1)

    #img_original_m_2 = a0.open_show_image(a0.file_chooser())
    #img_gray_m_2 = a0.grayscale(img_original_m_2)
    #imgFFTExchange(img_gray_m_1, img_gray_m_2)

    # 3 Ideal Filter
    #img_filtered_ideal = idealFilter(img_gray_m_1)
    #imageForms.showSideBySideImages(img_gray_m_1, img_filtered_ideal, 'Ideal')

    # 4 Gaussian Filter
    #img_filtered_gaussian = gaussianFilter(img_gray_m_1)
    #imageForms.showSideBySideImages(img_gray_m_1, img_filtered_gaussian, 'Gaussian')

    #imageForms.showSideBySideImages(img_filtered_ideal, img_filtered_gaussian, 'Ideal vs Gaussian')

    # 5 Focusing Application
    focusingApp()

