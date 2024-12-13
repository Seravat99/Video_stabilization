import cv2
import pyopencl as cl
import numpy as np


def kernel_setup_build() -> bool:
    try:
        platform = cl.get_platforms()
        my_gpu_devices = platform[0].get_devices(device_type=cl.device_type.GPU)  # select correct platform

        global ctx
        ctx = cl.Context(devices=my_gpu_devices)

        global commQ
        commQ = cl.CommandQueue(ctx)

        file = open("kernels_nao_mexer.c", "r")

        global prog
        prog = cl.Program(ctx, file.read())
        prog.build()

    except Exception as e:
        print(e)
        return False
    return True


def kernel_template_match_execute(template, gray_frame1, template_height, template_width, height, width):
    sad = np.zeros(shape=((gray_frame1.shape[1]-template_width+1)*(gray_frame1.shape[0]-template_height+1),), dtype=int)
    try:
        template_match = prog.template_match

        img_format = cl.ImageFormat(cl.channel_order.BGRA, cl.channel_type.UNSIGNED_INT8)

        memBuffer_frame1 = cl.Image(ctx, flags=cl.mem_flags.COPY_HOST_PTR | cl.mem_flags.READ_ONLY, format=img_format, shape=(gray_frame1.shape[1], gray_frame1.shape[0]), pitches=(gray_frame1.strides[0], gray_frame1.strides[1]), hostbuf=gray_frame1.data)
        memBuffer_template = cl.Image(ctx, flags=cl.mem_flags.COPY_HOST_PTR | cl.mem_flags.READ_ONLY, format=img_format, shape=(template.shape[1], template.shape[0]), pitches=(template.strides[0], template.strides[1]), hostbuf=template.data)
        memBuffer_sad = cl.Buffer(ctx, flags=cl.mem_flags.WRITE_ONLY, size=sad.nbytes)

        template_match.set_arg(0, memBuffer_frame1)
        template_match.set_arg(1, memBuffer_template)
        template_match.set_arg(2, memBuffer_sad)
        template_match.set_arg(3, np.int32(template_height))
        template_match.set_arg(4, np.int32(template_width))
        template_match.set_arg(5, np.int32(height))
        template_match.set_arg(6, np.int32(width))

        group_size = (width-template_width+1, height-template_height+1)
        local_size = (None)
        kernelEvent = cl.enqueue_nd_range_kernel(commQ, template_match, global_work_size=group_size, local_work_size=local_size)
        kernelEvent.wait()

        cl.enqueue_copy(commQ, sad, memBuffer_sad)

        memBuffer_frame1.release()
        memBuffer_template.release()
        memBuffer_sad.release()

        return sad

    except Exception as e:
        print(e)
    return sad


def kernel_translation_execute(frame1, y_mean, x_mean, height, width, template_height, template_width):
    new_frame = frame1.copy()
    try:
        template_match = prog.translation

        img_format = cl.ImageFormat(cl.channel_order.BGRA, cl.channel_type.UNSIGNED_INT8)

        memBuffer_frame1 = cl.Image(ctx, flags=cl.mem_flags.COPY_HOST_PTR | cl.mem_flags.READ_ONLY, format=img_format, shape=(frame1.shape[1], frame1.shape[0]), pitches=(frame1.strides[0], frame1.strides[1]), hostbuf=frame1.data)
        memBuffer_new_frame = cl.Image(ctx, flags=cl.mem_flags.COPY_HOST_PTR | cl.mem_flags.WRITE_ONLY, format=img_format, shape=(new_frame.shape[1], new_frame.shape[0]), pitches=(new_frame.strides[0], new_frame.strides[1]), hostbuf=new_frame.data)

        template_match.set_arg(0, memBuffer_frame1)
        template_match.set_arg(1, memBuffer_new_frame)
        template_match.set_arg(2, np.int32(y_mean))
        template_match.set_arg(3, np.int32(x_mean))
        template_match.set_arg(4, np.int32(height))
        template_match.set_arg(5, np.int32(width))
        template_match.set_arg(6, np.int32(template_height))
        template_match.set_arg(7, np.int32(template_width))

        group_size = (width, height)
        local_size = (None)
        kernelEvent = cl.enqueue_nd_range_kernel(commQ, template_match, global_work_size=group_size, local_work_size=local_size)
        kernelEvent.wait()

        cl.enqueue_copy(commQ, dest=new_frame, src=memBuffer_new_frame, origin=(0, 0, 0), region=(new_frame.shape[1], new_frame.shape[0]), is_blocking=True)

        memBuffer_frame1.release()
        memBuffer_new_frame.release()

        return new_frame

    except Exception as e:
        print(e)
    return new_frame


def development_function(cap):
    ret0, frame0 = cap.read()
    if not ret0:
        return

    height, width, _ = frame0.shape
    template_height = 31
    template_width = 31
    video = cv2.VideoWriter('video_stabilized.avi', cv2.VideoWriter_fourcc(*'DIVX'), 30, (width, height))
    video.write(frame0)

    while True:
        ret1, frame1 = cap.read()
        if ret1:
            bgra_frame0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2BGRA)
            bgra_frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2BGRA)

            # Create the template images
            start_x = int(width // 2 - template_width // 2)
            end_x = int(width // 2 + template_width // 2)
            start_y = int(height // 2 - template_height // 2)
            end_y = int(height // 2 + template_height // 2)
            template = np.ascontiguousarray(bgra_frame0[start_y:end_y, start_x:end_x])

            shift_values = kernel_template_match_execute(template, bgra_frame1, template_height, template_width, height, width)
            min_sad0 = np.min(shift_values)
            min_coord = np.where(shift_values == min_sad0)
            idx = min_coord[0][0]
            y_mean = start_y - idx // (width - template_width + 1)
            x_mean = start_x - idx % (width - template_width + 1)

            frame0 = kernel_translation_execute(bgra_frame1, y_mean*0.9, x_mean*0.9, height, width, template_height, template_width)
            video.write(cv2.cvtColor(frame0, cv2.COLOR_BGRA2BGR))
            cv2.imshow("stabilized frame", cv2.hconcat([cv2.cvtColor(frame0, cv2.COLOR_BGRA2BGR), frame1]))

            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break

        else:
            break
    video.release()


if __name__ == "__main__":
    video_directory = "..\\TAPDI_aulas\\images\\video_shake1.mp4"
    cap = cv2.VideoCapture(video_directory)

    kernel_build = kernel_setup_build()
    if kernel_build:
        development_function(cap)
    else:
        print("kernel not build")