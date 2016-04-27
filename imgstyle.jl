#!/usr/bin/julia

using MXNet, Images, DocOpt

include("stylenet.jl")
include("img_util.jl")

# TODO: Implement poor man's gradient checker. Hack the Python implementation
# to start with the same initial image as mine. Add I/O to serialize all
# gradients. Open blobs in REPL and check which ones are not equal.

# E.g. "1,2,3" -> (1,2,3)
str2inttup(str) = tuple(map(x -> parse(Int64, x), split(str, ','))...)

# E.g. "relu1_1,relu2_2" -> [relu1_1, relu2_2]
str2symbols(str) = map(Symbol, split(str, ','))

# Fixed Bug: doctopt strings must have >1 spaces between option names and
# description/default strings
usage = """IMGStyle.

Usage:
    imgstyle.jl [options] <content_img> <style_img>
    imgstyle.jl -h | --help

Options:
    -h, --help                       Show this screen.
    --output OUT_NAME                Provide a name for the output file. [default: out.png]
    --output_size OUT_SIZE
    --num_iter N_ITER                [default: 1000]
    --save_iter SAVE_ITER            [default: 100]
    --print_iter PRINT_ITER          [default: 10]
    --learning_rate LEARNING_RATE    [default: 0.1]
    --content_weight CONTENT_WEIGHT  [default: 0.5]
    --style_weight STYLE_WEIGHT      [default: 0.5]
    --content_layers CONTENT_LAYERS  [default: relu4_2]
    --style_layers STYLE_LAYERS      [default: relu1_1,relu2_1,relu3_1,relu4_1,relu5_1]
    --init_style
    --init_content
    --init_img INIT_IMG
"""
option_map = docopt(usage)

content_img = load(option_map["<content_img>"])
style_img = load(option_map["<style_img>"])

# Resize images for network output
out_size_str = option_map["--output_size"]
out_size = if out_size_str == nothing
    size(content_img)
else
    str2inttup(out_size_str)
end
content_img = Images.imresize(content_img, out_size)

# I think it's necessary to maintain the original aspect ratio of the style
# image to get the best result. Arbitrary resizing of the style image would
# change the style representation (assuming the style representation is not
# affine invariant). This is also a hack around the memory limitations of
# multiple executors in a garbage-collected language.
style_img = best_overlap(style_img, content_img)

# Get symbols for specified layers
style_layers = str2symbols(option_map["--style_layers"])
content_layers = str2symbols(option_map["--content_layers"])

# Construct and run the neural style network
stylenet = StyleNet(
    mx.gpu(), content_img, style_img, content_layers, style_layers)
output_img = optimize(stylenet)
save("output.png", output_img)
