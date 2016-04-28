using MXNet

include("vggnet.jl")
include("gramian.jl")

type StyleNet
    ctx :: mx.Context

    # Save memory using single executor
    exec :: mx.Executor

    img      :: mx.NDArray
    img_grad :: mx.NDArray

    content_repr :: Array{mx.NDArray,1} # static ReLU outputs
    content_grad :: Array{mx.NDArray,1} # L2 gradients
    content_out  :: Array{mx.NDArray,1} # transient ReLU outputs

    style_repr :: Array{mx.NDArray,1} # static Gramian matrices
    style_grad :: Array{mx.NDArray,1} # L2 gradients
    style_out  :: Array{mx.NDArray,1} # transient Gramian matrices

    # Shapes of inputs to Gramian layers
    style_shapes :: Array{Tuple,1}

    function StyleNet(ctx, content_img, style_img, content_layers, style_layers)
        content_arr = preprocess_vgg(content_img)
        content_size = size(content_arr)

        # Get symbolic nodes for loss from VGGNet layers
        content_nodes, style_nodes =
            vcat(content_layers, style_layers) |> make_vggnet
        content_group = mx.Group(content_nodes...)
        style_group = mx.Group(style_nodes...)

        # Get shapes of arguments and style ReLU outputs
        loss_group = mx.Group(content_group, style_group)
        arg_shapes, = mx.infer_shape(loss_group, img_data=content_size)
        ~, style_shapes, = mx.infer_shape(style_group, img_data=content_size)

        # Replace style layer outputs with Gramian outputs
        node = content_group
        for (i, layer) in enumerate(style_nodes)
            node = mx.Group(node, make_gramian(layer, style_shapes[i]))
        end

        # Allocate GPU memory for arguments and their gradients
        arg_names = mx.list_arguments(node)
        arg_map, grad_map =
            load_arguments(ctx, arg_names, arg_shapes, "model/vgg19.params")

        # Finalize network / make executor
        exec = mx.bind(node, ctx, arg_map, args_grad=grad_map)

        # Separate outputs for convenience
        num_content = size(content_layers, 1)
        content_out = exec.outputs[1:num_content]
        num_style = size(style_layers, 1)
        style_out = exec.outputs[num_content+1:end]

        # Allocate GPU memory for output gradients
        content_grad = map(x -> mx.zeros(size(x), ctx), content_out)
        style_grad = map(x -> mx.zeros(size(x), ctx), style_out)

        # Get ReLU output for content image
        img = arg_map[:img_data]
        img[:] = content_arr
        mx.forward(exec)
        content_repr = map(x -> copy(x, ctx), content_out)

        # Get Gramian matrices for style image
        img[:] = preprocess_vgg(style_img)
        mx.forward(exec)
        style_repr = map(x -> copy(x, ctx), style_out)

        # Initialize image to noise for optimization
        img[:] = mx.rand(-0.1, 0.1, size(content_arr))

        return new(ctx, exec, img, grad_map[:img_data], content_repr,
            content_grad, content_out, style_repr, style_grad, style_out,
            style_shapes)
    end
end

function optimize(net :: StyleNet)
    # Create SGD optimizer for image updates
    lr = mx.LearningRate.Fixed(0.1)
    sgd = mx.SGD(
        lr = 0.1,
        momentum = 0.9,
        weight_decay = 0.005,
        lr_scheduler = lr,
        grad_clip = 10)
    sgd_state = mx.create_state(sgd, 0, net.img)
    sgd.state = mx.OptimizationState(10)

    for epoch = 1:25
        mx.forward(net.exec)

        # Calculate output gradients
        num_content = size(net.content_repr, 1)
        num_style = size(net.style_repr, 1)
        for i, grad in enumerate(net.content_grad)
            grad[:] = net.content_out[i] - net.content_repr[i]
        end
        for i, grad in enumerate(net.style_grad)
            grad[:] = net.style_out[i] - net.style_repr[i]
            grad[:] /=
                (net.style_shapes[i][3]) * reduce(*, net.style_shapes[i][1:3])
        end

        mx.backward(net.exec, vcat(net.content_grad, net.style_grad))

        # Update image
        mx.update(sgd, 0, net.img, net.img_grad, sgd_state)
    end

    # Convert NDArray into Image
    return net.img |> copy |> postprocess_vgg
end

function load_arguments(
    ctx,
    arg_names :: Array{Symbol,1},
    arg_shapes :: Array{Tuple,1},
    model_path :: AbstractString)

    # Model is pre-trained. Get map from symbol to NDArray of weights or biases.
    pretrain = mx.load(model_path, mx.NDArray)

    # Zero-initialize arguments and gradients
    grad_map =
        Dict(zip(arg_names, [mx.zeros(shape, ctx) for shape in arg_shapes]))
    arg_map =
        Dict(zip(arg_names, [mx.zeros(shape, ctx) for shape in arg_shapes]))

    # Copy pre-trained weights and biases into new NDArrays
    for name in arg_names[2:end] # skip :img_data
        arg_map[name] = pretrain[symbol("arg:" * string(name))]
    end

    return (arg_map, grad_map)
end
