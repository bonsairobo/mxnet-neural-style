using MXNet, Images, Colors

# ImageNet mean pixel
mean_rgb = (103.939, 116.779, 123.68)

img2BGRarray(img) = convert(Image{BGR{Float32}}, img) |> separate |> data
BGRarray2img(arr) = convert(Image{BGR{Float32}}, clamp(arr / 256, 0, 1))

# VGG19 expects a specific image format
function preprocess_vgg(img)
    arr = 256 * img2BGRarray(img)
    arr[:,:,1] -= mean_rgb[3]
    arr[:,:,2] -= mean_rgb[2]
    arr[:,:,3] -= mean_rgb[1]
    return reshape(arr, (size(arr)...,1)) # Add batch dimension
end

# Undo preprocessing
function postprocess_vgg(arr)
    arr[:,:,1] += mean_rgb[3]
    arr[:,:,2] += mean_rgb[2]
    arr[:,:,3] += mean_rgb[1]
    return BGRarray2img(arr)
end

function make_vggnet(loss_symbols)
    # VGG model without fully-connected layers. Use AVG pooling for smoother
    # image optimization. Each layer has an explicit name to match a pretrained
    # model.
    img_data = mx.Variable(:img_data)
    conv1_1 = mx.Convolution(name=:conv1_1, data=img_data, num_filter=64,
        pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=false)
    relu1_1 = mx.Activation(name=:relu1_1, data=conv1_1, act_type=:relu)
    conv1_2 = mx.Convolution(name=:conv1_2, data=relu1_1, num_filter=64,
        pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=false)
    relu1_2 = mx.Activation(name=:relu1_2, data=conv1_2, act_type=:relu)
    pool1 = mx.Pooling(name=:pool1, data=relu1_2, pad=(0,0), kernel=(2,2),
        stride=(2,2), pool_type=:avg)
    conv2_1 = mx.Convolution(name=:conv2_1, data=pool1, num_filter=128,
        pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=false)
    relu2_1 = mx.Activation(name=:relu2_1, data=conv2_1, act_type=:relu)
    conv2_2 = mx.Convolution(name=:conv2_2, data=relu2_1, num_filter=128,
        pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=false)
    relu2_2 = mx.Activation(name=:relu2_2, data=conv2_2, act_type=:relu)
    pool2 = mx.Pooling(name=:pool2, data=relu2_2, pad=(0,0), kernel=(2,2),
        stride=(2,2), pool_type=:avg)
    conv3_1 = mx.Convolution(name=:conv3_1, data=pool2, num_filter=256,
        pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=false)
    relu3_1 = mx.Activation(name=:relu3_1, data=conv3_1, act_type=:relu)
    conv3_2 = mx.Convolution(name=:conv3_2, data=relu3_1, num_filter=256,
        pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=false)
    relu3_2 = mx.Activation(name=:relu3_2, data=conv3_2, act_type=:relu)
    conv3_3 = mx.Convolution(name=:conv3_3, data=relu3_2, num_filter=256,
        pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=false)
    relu3_3 = mx.Activation(name=:relu3_3, data=conv3_3, act_type=:relu)
    conv3_4 = mx.Convolution(name=:conv3_4, data=relu3_3, num_filter=256,
        pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=false)
    relu3_4 = mx.Activation(name=:relu3_4, data=conv3_4, act_type=:relu)
    pool3 = mx.Pooling(name=:pool3, data=relu3_4, pad=(0,0), kernel=(2,2),
        stride=(2,2), pool_type=:avg)
    conv4_1 = mx.Convolution(name=:conv4_1, data=pool3, num_filter=512,
        pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=false)
    relu4_1 = mx.Activation(name=:relu4_1, data=conv4_1, act_type=:relu)
    conv4_2 = mx.Convolution(name=:conv4_2, data=relu4_1, num_filter=512,
        pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=false)
    relu4_2 = mx.Activation(name=:relu4_2, data=conv4_2, act_type=:relu)
    conv4_3 = mx.Convolution(name=:conv4_3, data=relu4_2, num_filter=512,
        pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=false)
    relu4_3 = mx.Activation(name=:relu4_3, data=conv4_3, act_type=:relu)
    conv4_4 = mx.Convolution(name=:conv4_4, data=relu4_3, num_filter=512,
        pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=false)
    relu4_4 = mx.Activation(name=:relu4_4, data=conv4_4, act_type=:relu)
    pool4 = mx.Pooling(name=:pool4, data=relu4_4, pad=(0,0), kernel=(2,2),
        stride=(2,2), pool_type=:avg)
    conv5_1 = mx.Convolution(name=:conv5_1, data=pool4, num_filter=512,
        pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=false)
    relu5_1 = mx.Activation(name=:relu5_1, data=conv5_1, act_type=:relu)

    # I feel like I shouldn't have to do this...
    nodes = Dict(
        :conv1_1 => conv1_1,
        :relu1_1 => relu1_1,
        :conv1_2 => conv1_2,
        :relu1_2 => relu1_2,
        :pool1   => pool1,
        :conv2_1 => conv2_1,
        :relu2_1 => relu2_1,
        :conv2_2 => conv2_2,
        :relu2_2 => relu2_2,
        :pool2   => pool2,
        :conv3_1 => conv3_1,
        :relu3_1 => relu3_1,
        :conv3_2 => conv3_2,
        :relu3_2 => relu3_2,
        :conv3_3 => conv3_3,
        :relu3_3 => relu3_3,
        :conv3_4 => conv3_4,
        :relu3_4 => relu3_4,
        :pool3   => pool3,
        :conv4_1 => conv4_1,
        :relu4_1 => relu4_1,
        :conv4_2 => conv4_2,
        :relu4_2 => relu4_2,
        :conv4_3 => conv4_3,
        :relu4_3 => relu4_3,
        :conv4_4 => conv4_4,
        :relu4_4 => relu4_4,
        :pool4   => pool4,
        :conv5_1 => conv5_1,
        :relu5_1 => relu5_1
    )

    # Get nodes for loss symbols
    return map(s -> nodes[s], loss_symbols)
end
