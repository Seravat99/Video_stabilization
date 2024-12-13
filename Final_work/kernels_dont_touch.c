__constant sampler_t srcSampler = CLK_NORMALIZED_COORDS_FALSE |
                           CLK_ADDRESS_CLAMP_TO_EDGE |
                           CLK_FILTER_NEAREST ;

__kernel void template_match(__read_only image2d_t image, __read_only image2d_t template, __global int* output, int template_height, int template_width, int height, int width) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    int w = width-template_width+1;
    int idx = y * w + x;

    int p_imagex, p_imagey, p_imagez;
    int p_templatex, p_templatey, p_templatez;
    int diff = 0;
    int sum = 0;
    // loop through the template image
    for (int j = 0; j < template_height; j++){
        for (int i = 0; i < template_width; i++) {
            p_imagex = read_imageui(image, srcSampler, (int2)(x+i, y+j)).x;
            p_imagey = read_imageui(image, srcSampler, (int2)(x+i, y+j)).y;
            p_imagez = read_imageui(image, srcSampler, (int2)(x+i, y+j)).z;
            p_templatex = read_imageui(template, srcSampler, (int2)(i, j)).x;
            p_templatey = read_imageui(template, srcSampler, (int2)(i, j)).y;
            p_templatez = read_imageui(template, srcSampler, (int2)(i, j)).z;
            diff = (p_imagex - p_templatex) + (p_imagey - p_templatey) + (p_imagez - p_templatez);
            sum += abs(diff);
        }
    }
    output[idx] = sum;
}


__kernel void translation(__read_only image2d_t image, __write_only image2d_t image_output, int y_mean, int x_mean, int height, int width, int template_height, int template_width) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    int x_output = x - x_mean;
    int y_output = y - y_mean;

    if (x_output < 0 || y_output < 0 || x_output >= width || y_output >= height){
        write_imageui(image_output, (int2)(x, y), (uint4)(0,0,0,255));
    }
    else{
        uint4 color = read_imageui(image, srcSampler, (int2)(x_output, y_output));
        write_imageui(image_output, (int2)(x, y), color);
    }
}