#!/usr/bin/julia

using MXNet, Images, Colors, FixedPointNumbers, DocOpt

# Imagenet mean pixel
mean_rgb = (103.939, 116.779, 123.68)

# VGG19 expects a specific image format
function preprocess_vgg(img)
    arr = 256 * data(separate(convert(Image{BGR{Float32}}, img)))
    arr[:,:,1] -= mean_rgb[3]
    arr[:,:,2] -= mean_rgb[2]
    arr[:,:,3] -= mean_rgb[1]
    return arr
end

# Undo preprocessing
function postprocess_vgg(arr)
    arr[:,:,1] += mean_rgb[3]
    arr[:,:,2] += mean_rgb[2]
    arr[:,:,3] += mean_rgb[1]
    arr /= 256.0
    return convert(Image{RGB{Float32}}, arr)
end

# E.g. "1,2,3" -> (1,2,3)
str2inttup(str) = tuple(map(x -> parse(Int32, x), split(str, ','))...)

# E.g. "relu1_1,relu2_2" -> [relu1_1, relu2_2]
str2symbols(str) = map(mx.Variable, split(str, ','))

function make_gramian_executor()

end

function style_representation()

end

function content_representation()

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

# Determine output resolution
out_size_str = arg_map["--output_size"]
out_size = if out_size_str == nothing
    size(content_img)
else
    str2inttup(out_size_str)
end

# Only resize content image because the content loss function requires matching
# shape with output image. Style loss uses correlation between filters
# (Gramian), where shape is defined only by the # of filters.
content_img = Images.imresize(content_img, out_size)

content_arr = preprocess_vgg19(content_img)
style_arr = preprocess_vgg19(style_img)

# Create GPU-backed NDArrays for input images
content_nd = mx.copy(content_arr, gpu_context)
style_nd = mx.copy(style_arr, gpu_context)

# Get symbols for specified layers
style_layers = str2symbols(arg_map["--style_layers"])
content_layer = str2symbols(arg_map["--content_layer"])

# Create VGG19 executor
vgg_exec =
    make_vgg19_executor(out_size, style_layers, content_layer, gpu_context)
