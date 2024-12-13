# Video_stabilization
THIS IS FOR THE FINAL WORK. LABWORKS ARE VARIOUS EXERCISES

Stabilize a video with optical flow

TEMPLATE MATCH

In order to find out the number of pixels, in y and x, that varied from one frame to another, we decided to use the template match method. This method chooses a part of the first frame, with a variable dimension, in order to get the most out of it, and, using it as a template, looks for it in the second frame. In the chosen template, we decided to make a variable dimension, so that we could take better advantage of the frames. To do this, we had to create a program that compares the pixel values of the chosen template in the first frame with the pixel values of all the possible cases in the second frame and save the difference between each comparison. In order to make this process faster and more efficient, we implemented it in a kernel.
At the end of this process, we are left with all the possible differences between the two frames. As we want the best one, we choose the smallest one.

TRANSLATION

Having discovered the x and y values of the deviation that the camera made from one frame to the other, we made a kernel that applies 90% of this difference to the pixels of the second frame, subtracting it from the x and y coordinates. After applying the difference, the BGR values from the second frame are saved in a new image. The pixels that are found to be outside the original frame are given a black value in BGR. This new image will be the second frame stabilized in relation to the first.


#
Last time the code was tested -> 01/02/2022
