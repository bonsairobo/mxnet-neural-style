using MXNet

include("vggnet.jl")
include("layers.jl")

type StyleNet
    ctx :: mx.Context

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

    function StyleNet(ctx, content_img, style_img, content_layers, style_layers)
        style_arr = preprocess_vgg(style_img)
        content_arr = preprocess_vgg(content_img)

        num_content = size(content_layers, 1)
        num_style = size(style_layers, 1)

        # Make loss groups from VGGNet layers
        loss_nodes = make_vggnet(vcat(content_layers, style_layers))
        content_group = mx.Group(loss_nodes[1:num_content]...)
        loss_group = mx.Group(loss_nodes...)

        arg_shapes, out_shapes, =
            mx.infer_shape(loss_group, img_data=size(style_arr))

        # Replace style layer outputs with Gramian outputs
        node = content_group
        for (i, layer) in enumerate(loss_nodes[num_content+1:end])
            gram = make_gramian(layer, out_shapes[num_content + i])
            node = mx.Group(node, gram)
        end

        println("start with 0 GPU mem")
        sleep(3)

        # Allocate GPU memory for arguments and their gradients
        arg_names = mx.list_arguments(node)
        arg_map, grad_map =
            load_arguments!(ctx, arg_names, arg_shapes, "model/vgg19.params")

        println("after load args")
        sleep(3)

        gc() # Clean up unused zero NDArrays

        println("after gc()")
        sleep(3)

        net = new(ctx, node, arg_map, grad_map,
            [], fill(mx.zeros((0,), ctx), num_style),
            [], fill(mx.zeros((0,), ctx), num_content))

        println("after dummy mx.zeros, should not change mem usage")
        sleep(3)

        # img_data is already sized for style_arr from load_arguments
        net.arg_map[:img_data][:] = style_arr

        println("after NDArray assignment, should not change mem usage")
        sleep(3)

        exec = make_executor(net)

        println("after bind")
        sleep(3)
        
        # Get Gramian matrices for style image
        mx.forward(exec)
        for i = 1:num_style
            push!(net.style_repr, exec.outputs[num_content + i])
        end

        println("after copying output NDArrays, should not change mem usage")
        sleep(3)

        # Reset target shape in Reshape layer for content size by replacing
        # Gramian outputs
        arg_shapes, out_shapes, =
            mx.infer_shape(loss_group, img_data=size(content_arr))
        node = content_group
        for (i, layer) in enumerate(loss_nodes[num_content+1:end])
            gram = make_gramian(layer, out_shapes[num_content + i])
            node = mx.Group(node, gram)
        end
        net.node = node

        println("after symbolic Gramian creation, should not change mem usage")
        sleep(3)

        # Fit network shapes for the content image
        net.arg_map[:img_data] = mx.copy(content_arr, net.ctx)
        net.grad_map[:img_data] = mx.zeros(size(content_arr), net.ctx)

        println("after content image & gradient creation")
        sleep(3)

        exec = 0
        gc()

        println("after gc() of exec")
        sleep(3)

        exec = make_executor(net)

        println("after bind")
        sleep(3)
    
        # Get ReLU output for content image
        mx.forward(exec)
        for i = 1:num_content
            push!(net.content_repr, exec.outputs[i])
        end

        println("after copying output NDArrays, should not change mem usage")
        sleep(3)

        # Initialize data to noise for optimization
        net.arg_map[:img_data][:] = 100 * rand(size(content_arr))

        return net
    end
end

function update(net :: StyleNet)
    mx.forward(net.exec)

    # Calculate output gradients
    num_content = size(net.content_repr, 1)
    num_style = size(net.style_repr, 1)
    for i = 1:num_content
        net.content_grad[i] = l2_gradient(net.content_repr[i], net.outputs[i])
    end
    for i = 1:num_style
        net.style_grad[i] =
            l2_gradient(net.style_repr[i], net.outputs[num_content + i])
    end
    net.tv_grad = tv_gradient(net.arg_map[:data])

    mx.backward(net.exec, vcat(net.content_repr, net.style_repr))
end

function load_arguments!(
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

function make_executor(net :: StyleNet)
    # Create new executor with different input/output shapes
    return mx.bind(net.node, net.ctx, net.arg_map, args_grad=net.grad_map)
end
