using MXNet

type VGG_Executor
    executor :: mx.Executor

    # Provided in forward pass
    data :: mx.NDArray

    # Updated in forward pass
    style_activations   :: Array{mx.NDArray,1}
    content_activations :: mx.NDArray

    # Updated in backward pass (requires gradients from all activations)
    data_grad :: mx.NDArray
end

function make_vgg19_executor(input_size, style_layers, content_layer, context)
    # VGG model without fully-connected layers. Use AVG pooling for smoother
    # image optimization.
    data = mx.Variable(:data)
    conv1_1 = mx.Convolution(name=:conv1_1, data=data, num_filter=64, pad=(1,1),
        kernel=(3,3), stride=(1,1), no_bias=false, workspace=1024)
    relu1_1 = mx.Activation(name=:relu1_1, data=conv1_1, act_type=:relu)
    conv1_2 = mx.Convolution(name=:conv1_2, data=relu1_1, num_filter=64,
        pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=false, workspace=1024)
    relu1_2 = mx.Activation(name=:relu1_2, data=conv1_2, act_type=:relu)
    pool1 = mx.Pooling(name=:pool1, data=relu1_2, pad=(0,0), kernel=(2,2),
        stride=(2,2), pool_type=:avg)
    conv2_1 = mx.Convolution(name=:conv2_1, data=pool1, num_filter=128,
        pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=false, workspace=1024)
    relu2_1 = mx.Activation(name=:relu2_1, data=conv2_1, act_type=:relu)
    conv2_2 = mx.Convolution(name=:conv2_2, data=relu2_1, num_filter=128,
        pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=false, workspace=1024)
    relu2_2 = mx.Activation(name=:relu2_2, data=conv2_2, act_type=:relu)
    pool2 = mx.Pooling(name=:pool2, data=relu2_2, pad=(0,0), kernel=(2,2),
        stride=(2,2), pool_type=:avg)
    conv3_1 = mx.Convolution(name=:conv3_1, data=pool2, num_filter=256,
        pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=false, workspace=1024)
    relu3_1 = mx.Activation(name=:relu3_1, data=conv3_1, act_type=:relu)
    conv3_2 = mx.Convolution(name=:conv3_2, data=relu3_1, num_filter=256,
        pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=false, workspace=1024)
    relu3_2 = mx.Activation(name=:relu3_2, data=conv3_2, act_type=:relu)
    conv3_3 = mx.Convolution(name=:conv3_3, data=relu3_2, num_filter=256,
        pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=false, workspace=1024)
    relu3_3 = mx.Activation(name=:relu3_3, data=conv3_3, act_type=:relu)
    conv3_4 = mx.Convolution(name=:conv3_4, data=relu3_3, num_filter=256,
        pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=false, workspace=1024)
    relu3_4 = mx.Activation(name=:relu3_4, data=conv3_4, act_type=:relu)
    pool3 = mx.Pooling(name=:pool3, data=relu3_4, pad=(0,0), kernel=(2,2),
        stride=(2,2), pool_type=:avg)
    conv4_1 = mx.Convolution(name=:conv4_1, data=pool3, num_filter=512,
        pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=false, workspace=1024)
    relu4_1 = mx.Activation(name=:relu4_1, data=conv4_1, act_type=:relu)
    conv4_2 = mx.Convolution(name=:conv4_2, data=relu4_1, num_filter=512,
        pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=false, workspace=1024)
    relu4_2 = mx.Activation(name=:relu4_2, data=conv4_2, act_type=:relu)
    conv4_3 = mx.Convolution(name=:conv4_3, data=relu4_2, num_filter=512,
        pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=false, workspace=1024)
    relu4_3 = mx.Activation(name=:relu4_3, data=conv4_3, act_type=:relu)
    conv4_4 = mx.Convolution(name=:conv4_4, data=relu4_3, num_filter=512,
        pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=false, workspace=1024)
    relu4_4 = mx.Activation(name=:relu4_4, data=conv4_4, act_type=:relu)
    pool4 = mx.Pooling(name=:pool4, data=relu4_4, pad=(0,0), kernel=(2,2),
        stride=(2,2), pool_type=:avg)
    conv5_1 = mx.Convolution(name=:conv5_1, data=pool4, num_filter=512,
        pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=false, workspace=1024)
    relu5_1 = mx.Activation(name=:relu5_1, data=conv5_1, act_type=:relu)

    # Group loss layers for convenient access to all unbound arguments
    # (data, weights, biases) on which the loss is dependent
    style_group = mx.Group(style_layers...)
    content_group = mx.Group(content_layer...)
    loss_group = mx.Group(style_group, content_group)

    # Infer argument shapes. Data batch size = 1, and data shape is given in
    # Julia-native column-major order.
    arg_shapes, =
        mx.infer_shape(loss_group, data=(input_size[2],input_size[1],3,1))
    arg_names = mx.list_arguments(loss_group)

    # Zero-initialize arguments and gradients
    grad_nd_map =
        Dict(zip(arg_names, [mx.zeros(shape) for shape in arg_shapes]))
    arg_nd_map = Dict(zip(arg_names, [mx.zeros(shape) for shape in arg_shapes]))

    # Model is pre-trained. Get map from symbol to NDArray of weights or biases.
    pretrain_nd_map = mx.load("model/vgg19.params", mx.NDArray)

    # Copy pre-trained weights and biases into new NDArrays
    for name in arg_names[2:end] # skip :data
        mx.copy!(
            arg_nd_map[name], pretrain_nd_map[symbol("arg:" * string(name))])
    end

    # Bind arguments and gradients to a group to create an executor. This
    # executor has one output for every layer in the loss group (ordered by the
    # group composition).
    exec = mx.bind(loss_group, context, arg_nd_map, args_grad=grad_nd_map)

    # Keep references to arrays used by executor
    return VGG_Executor(
        exec,
        arg_nd_map[:data],
        exec.outputs[1:end-1],
        exec.outputs[end],
        grad_nd_map[:data])
end
