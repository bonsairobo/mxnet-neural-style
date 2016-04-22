#!/usr/bin/julia

using MXNet, Images, DocOpt

include("stylenet.jl")

# E.g. "1,2,3" -> (1,2,3)
str2inttup(str) = tuple(map(x -> parse(Int32, x), split(str, ','))...)

# E.g. "relu1_1,relu2_2" -> [relu1_1, relu2_2]
str2symbols(str) = map(mx.Variable, split(str, ','))

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
    --content_layers CONTENT_LAYERS [default: relu4_2]
    --style_layers STYLE_LAYERS     [default: relu1_1,relu2_1,relu3_1,relu4_1,relu5_1]
    --init_style
    --init_content
    --init_img INIT_IMG
"""
option_map = docopt(usage)

content_img = imread(option_map["<content_image>"])
style_img = imread(option_map["<style_image>"])

# Determine output resolution
out_size_str = option_map["--output_size"]
out_size = if out_size_str == nothing
    size(content_img)
else
    str2inttup(out_size_str)
end

# Only resize content image because the content loss function requires matching
# shape with output image. Style loss uses correlation between filters
# (Gramian), where shape is defined only by the # of filters.
content_img = Images.imresize(content_img, out_size)

# Get symbols for specified layers
style_layers = str2symbols(option_map["--style_layers"])
content_layers = str2symbols(option_map["--content_layers"])

stylenet = StyleNet(
    mx.gpu(), content_img, style_img, content_layers, style_layers)

while

end
