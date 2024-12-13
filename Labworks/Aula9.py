import pyopencl as cl
import imageForms as iF
import numpy as np
import cv2 as cv
import imageForms as iF
import math


def SetupAndBuild():
    try:
        platforms = cl.get_platforms()

        global platform
        platform = platforms[1]
        devices = platform.get_devices()

        global device
        device = devices[0]

        global ctx
        ctx = cl.Context(devices)  # or dev_type=cl.device_type.ALL)

        global commQ
        commQ = cl.CommandQueue(ctx, device)

        file = open("kernel.c", "r")
        global prog
        prog = cl.Program(ctx, file.read())
        prog.build()
    except Exception as e:
        print(e)
        return False
    return True


def MemoryAndParameters():
    try:

        brightness = 10
        contrast = 5

        img = cv.imread("..\\TAPDI_aulas\\images\\road (4).bmp")
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img_out = np.zeros_like(img)
        img_neg = abs(255 - img)
        width = img.shape[1]
        height = img.shape[0]
        padding = img.strides[0] - width * img.strides[1] * img.itemsize

        bufferMem = cl.Buffer(ctx, flags=cl.mem_flags.COPY_HOST_PTR | cl.mem_flags.READ_WRITE, size=img.nbytes, hostbuf=img.data)
        bufferOut = cl.Buffer(ctx, flags=cl.mem_flags.COPY_HOST_PTR | cl.mem_flags.READ_WRITE, size=img_out.nbytes, hostbuf=img_out.data)

        kernelName = prog.negative_image2D

        kernelName.set_arg(0, bufferMem)
        kernelName.set_arg(1, np.int32(width))
        kernelName.set_arg(2, np.int32(height))
        kernelName.set_arg(3, np.int32(padding))
        kernelName.set_arg(4, bufferOut)
        # kernelName.set_arg(4, brightness)
        # kernelName.set_arg(5, contrast)

        x = math.ceil(width/32) * 32
        y = math.ceil(height/32) * 32
        workGroupSize = (x, y)
        workItemSize = (32, 32)     #
        kernelEvent = cl.enqueue_nd_range_kernel(commQ, kernelName,
                                                 global_work_size=workGroupSize, local_work_size=workItemSize)
        kernelEvent.wait()

        #cl.enqueue_copy(commQ, dest=bufferOut, src=bufferMem, origin=img, region=img, is_blocking=True)
        cl.enqueue_copy(commQ, dest=img.data, src=bufferMem)
        cl.enqueue_copy(commQ, dest=img_out.data, src=bufferOut)

        print(img_neg)
        print("img_out")
        print(img_out)
        print(img_neg == img_out)
        iF.showImage(img_out,"out")
        iF.showImage(img_neg,"neg")

        iF.showSideBySideImages(img_neg, img_out, False, True)
        cv.wait()
        bufferMem.release()
    except Exception as e:
        print(e)
        return False
    return True


def ex1():
    SetupAndBuild()
    MemoryAndParameters()


if __name__ == "__main__":
    exercise = int(input('What is the exercise?\n'))

    if exercise == 1:
        ex1()


