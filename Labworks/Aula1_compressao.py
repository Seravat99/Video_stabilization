from Aula0 import*

import huffman as h
import matplotlib
from matplotlib import pyplot


def histogram(img):

    hist = cv.calcHist([img], [0], None, [256], [0, 256])
    pyplot.plot(hist), pyplot.xlim([0, 256])

    for x in range(0, len(hist)):
        val = hist[x]
        if val > 0:
          print('->' + str(x) + ' - ' + str(int(val)))
    pyplot.show()
    cv.waitKey()


def huffman(img):
    huffman_dict = h.huffman(img.flatten())
    print(huffman_dict)


# histogram()
def main():
    file_path = file_chooser()
    img = open_show_image(file_path)
    histogram(img)
    huffman(img)

main()
