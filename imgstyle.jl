#!/usr/bin/julia

using MXNet, Images, DocOpt

include("stylenet.jl")

# E.g. "1,2,3" -> (1,2,3)
str2inttup(str) = tuple(map(x -> parse(Int64, x), split(str, ','))...)

# E.g. "relu1_1,relu2_2" -> [relu1_1, relu2_2]
str2symbols(str) = map(Symbol, split(str, ','))

overlap_area(size1, size2) = min(size1[1], size2[1]) * min(size1[2], size2[2])

function best_overlap(img1, img2)
    size1 = size(img1)
    size2 = size(img2)

    # Check if img1 already covers img2
    if size1[1] >= size2[1] && size1[2] >= size2[2]
        # Cut img2-sized region from img1 (no need to modify the scale)
        img1 = convert(Image, transpose(img1[1:size2[1], 1:size2[2]]))
        return img1
    end

    # Transpose img1 if it has greater overlap
    overlap = overlap_area(size1, size2)
    flip_overlap = overlap_area(reverse(size1), size2)
    if flip_overlap > overlap
        img1 = convert(Image, img1[:,:])
        size1 = size(img1)
    end

    # Check again if img1 covers img2
    if size1[1] >= size2[1] && size1[2] >= size2[2]
        # Cut img2-sized piece from img1 (no need to modify the scale)
        img1 = convert(Image, transpose(img1[1:size2[1], 1:size2[2]]))
        return img1
    end

    # Scale img1 to cover img2
    scalex = size2[1] / size1[1]
    scaley = size2[2] / size1[2]
    scale = max(scalex, scaley)
    new_size = map(x -> Int64(ceil(scale*x)), size1)
    img1 = Images.imresize(img1, new_size)

    # Crop img1 to the same size as img2
    img1 = convert(Image, transpose(img1[1:size2[1], 1:size2[2]]))
    return img1
end

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
style_img = best_overlap(style_img, content_img)

# Get symbols for specified layers
style_layers = str2symbols(option_map["--style_layers"])
content_layers = str2symbols(option_map["--content_layers"])

stylenet = StyleNet(
    mx.gpu(), content_img, style_img, content_layers, style_layers)
