#!/usr/bin/julia

using MXNet, Images, DocOpt

include("stylenet.jl")

# E.g. "1,2,3" -> (1,2,3)
str2inttup(str) = tuple(map(x -> parse(Int64, x), split(str, ','))...)

# E.g. "relu1_1,relu2_2" -> [relu1_1, relu2_2]
str2symbols(str) = map(Symbol, split(str, ','))

overlap_area(size1, size2) = reduce(*, min(size1, size2))

function crop_center(img, crop_size)
    imsize = img |> size |> collect
    tl = Array{Int64}(floor((imsize - crop_size) / 2) + 1)
    br = tl + crop_size - 1
    return copyproperties(img, img[tl[1]:br[1], tl[2]:br[2]])
end

function best_overlap(img1, img2)
    size1 = img1 |> size |> collect
    size2 = img2 |> size |> collect

    # Check if img1 already covers img2
    if all(size1 .> size2)
        return copyproperties(img1, crop_center(img1, size2))
    end

    # Transpose img1 if it has greater overlap
    overlap = overlap_area(size1, size2)
    flip_overlap = overlap_area(reverse(size1), size2)
    if flip_overlap > overlap
        img1 = copyproperties(img1, transpose(img1[:,:]))
        size1 = img1 |> size |> collect
    end

    # Check again if img1 covers img2
    if all(size1 .> size2)
        return copyproperties(img1, crop_center(img1, size2))
    end

    # Scale img1 to cover img2
    scale = maximum(size2 ./ size1)
    new_size = map(x -> Int64(ceil(scale * x)), size1)
    img1 = Images.imresize(img1, (new_size...))

    # Crop img1 (centered) to the same size as img2
    return copyproperties(img1, crop_center(img1, size2))
end

usage = """IMGStyle.

Usage:
    imgstyle.jl [options] <content_img> <style_img>
    imgstyle.jl -h | --help

Options:
    -h, --help                       Show this screen.
    --output OUT_NAME                Provide a name for the output file. [default: out.png]
    --long_edge LONG_EDGE
    --num_iter N_ITER                [default: 500]
    --save_iter SAVE_ITER            [default: 100]
    --content_weight CONTENT_WEIGHT  [default: 0.5]
    --style_weight STYLE_WEIGHT      [default: 0.5]
    --content_layers CONTENT_LAYERS  [default: relu4_2]
    --style_layers STYLE_LAYERS      [default: relu1_1,relu2_1,relu3_1,relu4_1,relu5_1]
"""
option_map = docopt(usage)

content_img = load(option_map["<content_img>"])
style_img = load(option_map["<style_img>"])

# Resize images for network output
long_edge_str = option_map["--long_edge"]
out_size = if long_edge_str == nothing
    size(content_img)
else
    orig = content_img |> size |> collect
    tuple(map(x -> Int64(round(x)), (parse(long_edge_str) / maximum(orig)) * orig)...)
end
content_img = Images.imresize(content_img, out_size)

# I think it's necessary to maintain the original aspect ratio of the style
# image to get the best result. Arbitrary resizing of the style image would
# change the style representation. This is also a hack around the memory
# limitations of reusing executors in a garbage-collected language.
# WARNING: This requires high image resolution for the best result.
style_img = best_overlap(style_img, content_img)
#style_img = Images.imresize(style_img, out_size)

save("style_in.png", style_img)

# Get symbols for specified layers
style_layers = str2symbols(option_map["--style_layers"])
content_layers = str2symbols(option_map["--content_layers"])

# Construct and run the neural style network
stylenet = StyleNet(
    mx.gpu(), content_img, style_img, content_layers, style_layers)
output_img = optimize(stylenet, option_map)
save("output.png", output_img)

# Despeckle a few times
for i = 1:4
    run(`convert output.png -despeckle output.png`)
end
