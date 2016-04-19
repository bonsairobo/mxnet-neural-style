#!/usr/bin/julia

function preprocess_vgg19()

end

function postprocess_vgg19()

end

function imgstyle(arg_map)

end

using DocOpt

usage = """IMGStyle.

Usage:
    imgstyle.jl [options] <content_img> <style_img> 
    imgstyle.jl -h | --help

Options:
    -h, --help                      Show this screen.
    --output OUT_NAME               Provide a name for the output file. [default: out.png]
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

imgstyle(arg_map)
