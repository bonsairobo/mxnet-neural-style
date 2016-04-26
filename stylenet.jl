using MXNet
using Debug

include("vggnet.jl")
include("layers.jl")

type StyleNet
    ctx :: mx.Context
    exec :: mx.Executor

    # All output (loss) nodes & arguments in the network
    node :: mx.SymbolicNode

    # Arguments and their gradients (provided to create Executor)
    arg_map :: Dict{Symbol, mx.NDArray}
    grad_map :: Dict{Symbol, mx.NDArray}

    # Parallel arrays
    style_repr :: Array{mx.NDArray,1} # Gramian matrices
    style_grad :: Array{mx.NDArray,1} # L2 gradients
    content_repr :: Array{mx.NDArray,1} # ReLU outputs
    content_grad :: Array{mx.NDArray,1} # L2 gradients

    # Total Variation gradient for image
    tv_grad :: mx.NDArray

    # Needed for style gradient normalization
    style_out_shapes :: Array{Tuple,1}

    @debug function StyleNet(ctx, content_img, style_img, content_layers, style_layers)
        style_arr = preprocess_vgg(style_img)
        content_arr = preprocess_vgg(content_img)

        num_content = size(content_layers, 1)
        num_style = size(style_layers, 1)

        # Get symbolic nodes for loss from VGGNet layers
        loss_nodes = vcat(content_layers, style_layers) |> make_vggnet

        arg_shapes, out_shapes, =
            mx.infer_shape(mx.Group(loss_nodes...), img_data=size(content_arr))

        # Replace style layer outputs with Gramian outputs
        node = mx.Group(loss_nodes[1:num_content]...)
        for (i, layer) in enumerate(loss_nodes[num_content+1:end])
            gram = make_gramian(layer, out_shapes[num_content + i])
            node = mx.Group(node, gram)
        end

        # Allocate GPU memory for arguments and their gradients
        arg_names = mx.list_arguments(node)
        arg_map, grad_map =
            load_arguments(ctx, arg_names, arg_shapes, "model/vgg19.params")

        # Finalize network / make executor
        exec = mx.bind(node, ctx, arg_map, args_grad=grad_map)

        # Allocate GPU memory for output gradients
        style_grad =
            map(x -> mx.zeros(size(x), ctx), exec.outputs[num_content+1:end])
        content_grad =
            map(x -> mx.zeros(size(x), ctx), exec.outputs[1:num_content])

        # Get Gramian matrices for style image
        arg_map[:img_data][:] = style_arr
        mx.forward(exec)
        style_repr = exec.outputs[num_content+1:end]
    
        # Get ReLU output for content image
        arg_map[:img_data][:] = content_arr
        mx.forward(exec)
        content_repr = exec.outputs[1:num_content]

        # Initialize data to noise for optimization
        init = 200 * (rand(content_arr) - 0.5)
        arg_map[:img_data][:] = init
        @bp

        return new(ctx, exec, node, arg_map, grad_map,
            style_repr, style_grad, content_repr, content_grad,
            mx.zeros((0,), ctx), out_shapes)
    end
end

function optimize(net :: StyleNet)
    lr = mx.LearningRate.Exp(0.1)
    sgd = mx.SGD(
        lr = 0.1,
        momentum = 0.9,
        weight_decay = 0.005,
        lr_scheduler = lr,
        grad_clip = 10)
    sgd_state = mx.create_state(sgd, 0, net.arg_map[:img_data])
    sgd.state = mx.OptimizationState(1)

    for epoch = 1:1
        mx.forward(net.exec)

        # Calculate output gradients
        num_content = size(net.content_repr, 1)
        num_style = size(net.style_repr, 1)
        for i = 1:num_content
            net.content_grad[i][:] = net.exec.outputs[i] - net.content_repr[i]
        end
        for i = 1:num_style
            net.style_grad[i][:] =
                net.exec.outputs[num_content+i] - net.style_repr[i]
            net.style_grad[i][:] /=
                (net.style_out_shapes[num_content+i][3] ^ 2) *
                reduce(*, net.style_out_shapes[num_content+i][1:2])
        end
        #net.tv_grad = tv_gradient(net.arg_map[:data])

        mx.backward(net.exec, vcat(net.content_grad, net.style_grad))

        # Update image
        mx.update(
            sgd, 0, net.arg_map[:img_data], net.grad_map[:img_data], sgd_state)
    end

    # Convert NDArray into Image
    out_arr = net.arg_map[:img_data] |> size |> zeros
    copy!(out_arr, net.arg_map[:img_data])
    return postprocess_vgg(out_arr)
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
    for name in arg_names[2:end] # skip :data
        arg_map[name] = pretrain[symbol("arg:" * string(name))]
    end

    return (arg_map, grad_map)
end
