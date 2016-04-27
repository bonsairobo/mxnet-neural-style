using Images

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
        return copyproperties(img1, crop_center(im1, size2))
    end

    # Scale img1 to cover img2
    scale = maximum(size2 ./ size1)
    new_size = map(x -> Int64(ceil(scale * x)), size1)
    img1 = Images.imresize(img1, (new_size...))

    # Crop img1 (centered) to the same size as img2
    return copyproperties(img1, crop_center(img1, size2))
end
