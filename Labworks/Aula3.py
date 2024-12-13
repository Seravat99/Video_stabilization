
from Aula0 import*
import ImageSegmentation
import imageForms
import cv2 as cv

if __name__ == '__main__':

    file_path = file_chooser()
    img = open_show_image(file_path)
    img_orig = img.copy()

    imgLabels = ImageSegmentation.GetConnectedComponents(img)
    img[imgLabels == 0] = [255, 0, 0]

    img_segmented = ImageSegmentation.Kmeans_Clustering(img_orig, 10)

    #img_neg = 255 - img_orig
    file_path = file_chooser()
    img_with_marks = open_show_image(file_path)

    img_marks = ImageSegmentation.GetWatershedFromMarks(img_orig, img_with_marks)

    #imgLabels_marks = ImageSegmentation.GetConnectedComponents(img_marks)
    #img_with_marks[imgLabels_marks == 0] = [255, 0, 0]

    #img_imm = ImageSegmentation.GetWatershedByImmersion(img_orig)
    imageForms.showSideBySideImages(img_orig, img_with_marks, BGR1=False, BGR2=False)

    cv.waitKey()
