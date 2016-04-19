#!/usr/bin/julia

using Images, Colors, FixedPointNumbers, DocOpt

# Imagenet mean pixel
mean_rgb = (103.939, 116.779, 123.68)

# VGG19 expects a specific image format
function preprocess_vgg(img, out_size)
    img = Images.imresize(img, out_size)
    arr = 256 * data(separate(convert(Image{BGR{Float32}}, img)))
    arr[:,:,1] -= mean_rgb[3]
    arr[:,:,2] -= mean_rgb[2]
    arr[:,:,3] -= mean_rgb[1]
    return arr
end

# undo preprocessing
function postprocess_vgg(arr)
    arr[:,:,1] += mean_rgb[3]
    arr[:,:,2] += mean_rgb[2]
    arr[:,:,3] += mean_rgb[1]
    arr /= 256.0
    return convert(Image, arr)
end

usage = """IMGStyle.

Usage:
    imgstyle.jl [options] <content_img> <style_img> 
    imgstyle.jl -h | --help

Options:
    -h, --help                      Show this screen.
    --output OUT_NAME               Provide a name for the output file. [default: out.png]
    --output_size OUT_SIZE
    --num_iter N_ITER               [default: 1000]
    --save_iter SAVE_ITER           [default: 100]
    --print_iter PRINT_ITER         [default: 10]
    --learning_rate LEARNING_RATE   [default: 0.1]
    --style_weight STYLE_WEIGHT     [default: 0.5]
    --content_weight CONTENT_WEIGHT [default: 0.5]
    --content_layer CONTENT_LAYER   [default: relu4_2]
    --style_layers STYLE_LAYERS     [default: relu1_1,relu2_1,relu3_1,relu4_1,relu5_1]
    --init_style
    --init_content
    --init_img INIT_IMG
"""
arg_map = docopt(usage)

gpu_context = mx.gpu()

content_img = imread(arg_map["<content_image>"])
style_img = imread(arg_map["<style_image>"])

# determine output resolution
out_size_str = arg_map["--output_size"]
out_size = if out_size_str != nothing
    split(out_size_str, ',')
else
    size(content_img)
end

content_arr = preprocess_vgg19(content_img, out_size)
style_arr = preprocess_vgg19(style_img, out_size)

# create GPU-backed NDArrays
content_nd = mx.copy(content_arr, gpu_context)
style_nd = mx.copy(style_arr, gpu_context)
