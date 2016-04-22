using MXNet, Images, Colors

# ImageNet mean pixel
mean_rgb = (103.939, 116.779, 123.68)

img2BGRarray(img) = data(separate(convert(Image{BGR{Float32}}, img)))

# VGG19 expects a specific image format
function preprocess_vgg(img)
    arr = 256 * img2BGRarray(img)
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
    arr /= 256
    return convert(Image{BGR{Float32}}, arr)
end

function make_vggnet()
    # VGG model without fully-connected layers. Use AVG pooling for smoother
    # image optimization. Each layer has an explicit name to match a pretrained
    # model.
    @mx.chain mx.Variable(:data)                                        =>
        mx.Convolution(name=:conv1_1, num_filter=64, pad=(1,1), kernel=(3,3),
            stride=(1,1), no_bias=false, workspace=1024)                =>
        mx.Activation(name=:relu1_1, act_type=:relu)                    =>
        mx.Convolution(name=:conv1_2, num_filter=64, pad=(1,1), kernel=(3,3),
            stride=(1,1), no_bias=false, workspace=1024)                =>
        mx.Activation(name=:relu1_2, act_type=:relu)                    =>
        mx.Pooling(name=:pool1, pad=(0,0), kernel=(2,2), stride=(2,2),
            pool_type=:avg)                                             =>
        mx.Convolution(name=:conv2_1, num_filter=128, pad=(1,1), kernel=(3,3),
            stride=(1,1), no_bias=false, workspace=1024)                =>
        mx.Activation(name=:relu2_1, act_type=:relu)                    =>
        mx.Convolution(name=:conv2_2, num_filter=128, pad=(1,1), kernel=(3,3),
            stride=(1,1), no_bias=false, workspace=1024)                =>
        mx.Activation(name=:relu2_2, act_type=:relu)                    =>
        mx.Pooling(name=:pool2, pad=(0,0), kernel=(2,2), stride=(2,2),
            pool_type=:avg)                                             =>
        mx.Convolution(name=:conv3_1, num_filter=256, pad=(1,1), kernel=(3,3),
            stride=(1,1), no_bias=false, workspace=1024)                =>
        mx.Activation(name=:relu3_1, act_type=:relu)                    =>
        mx.Convolution(name=:conv3_2, num_filter=256, pad=(1,1), kernel=(3,3),
            stride=(1,1), no_bias=false, workspace=1024)                =>
        mx.Activation(name=:relu3_2, act_type=:relu)                    =>
        mx.Convolution(name=:conv3_3, num_filter=256, pad=(1,1), kernel=(3,3),
            stride=(1,1), no_bias=false, workspace=1024)                =>
        mx.Activation(name=:relu3_3, act_type=:relu)                    =>
        mx.Convolution(name=:conv3_4, num_filter=256, pad=(1,1), kernel=(3,3),
            stride=(1,1), no_bias=false, workspace=1024)                =>
        mx.Activation(name=:relu3_4, act_type=:relu)                    =>
        mx.Pooling(name=:pool3, pad=(0,0), kernel=(2,2), stride=(2,2),
            pool_type=:avg)                                             =>
        mx.Convolution(name=:conv4_1, num_filter=512, pad=(1,1), kernel=(3,3),
            stride=(1,1), no_bias=false, workspace=1024)                =>
        mx.Activation(name=:relu4_1, act_type=:relu)                    =>
        mx.Convolution(name=:conv4_2, num_filter=512, pad=(1,1),
            kernel=(3,3), stride=(1,1), no_bias=false, workspace=1024)  =>
        mx.Activation(name=:relu4_2, act_type=:relu)                    =>
        mx.Convolution(name=:conv4_3, num_filter=512, pad=(1,1), kernel=(3,3),
            stride=(1,1), no_bias=false, workspace=1024)                =>
        mx.Activation(name=:relu4_3, act_type=:relu)                    =>
        mx.Convolution(name=:conv4_4, num_filter=512, pad=(1,1), kernel=(3,3),
            stride=(1,1), no_bias=false, workspace=1024)                =>
        mx.Activation(name=:relu4_4, act_type=:relu)                    =>
        mx.Pooling(name=:pool4, pad=(0,0), kernel=(2,2), stride=(2,2),
            pool_type=:avg)                                             =>
        mx.Convolution(name=:conv5_1, num_filter=512, pad=(1,1), kernel=(3,3),
            stride=(1,1), no_bias=false, workspace=1024)                =>
        mx.Activation(name=:relu5_1, act_type=:relu)
end
