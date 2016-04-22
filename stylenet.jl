using MXNet

include("vggnet.jl")

type StyleNet
    ctx :: mx.Context

    # All output (loss) nodes & arguments in the network
    node :: mx.SymbolicNode

    # Executor for back/forward propagation
    exec

    # Arguments and their gradients (provided to create Executor)
    arg_map :: Dict{Symbol, mx.NDArray}
    grad_map :: Dict{Symbol, mx.NDArray}

    # Parallel arrays
    style_layers :: Array{Symbol,1}
    style_repr :: Array{mx.NDArray,1} # Gramian matrices
    style_grad :: Array{mx.NDArray,1} # L2 gradients

    content_layers :: Array{Symbol,1}
    content_repr :: Array{mx.NDArray,1} # ReLU outputs
    content_grad :: Array{mx.NDArray,1} # L2 gradients

    # Total Variation gradient for image
    tv_grad :: mx.NDArray

    function StyleNet(ctx, content_img, style_img, content_layers, style_layers)
        style_arr = preprocess_vgg(style_img)
        content_arr = preprocess_vgg(content_img)
        style_data_size = (size(style_arr)...,1)
        content_data_size = (size(content_arr)...,1)

        # Initialize symbolic graph with VGG CNN
        make_vggnet()
        node = mx.Group(content_layers..., style_layers...)
        arg_shapes, out_shapes, = mx.infer_shape(node, data=style_data_size)

        # Replace style layer outputs with Gramian outputs
        node = mx.Group(content_layers...)
        num_content = size(content_layers, 1)
        for (i, layer) in enumerate(style_layers)
            gram = make_gramian(layer, out_shapes[num_content + i])
            node = mx.Group(node, gram)
        end

        # Allocate GPU memory for arguments and their gradients
        arg_map = Dict()
        grad_map = Dict()
        arg_names = mx.list_arguments(node)
        load_arguments!(
            arg_names, arg_shapes, arg_map, grad_map, "model/vgg19.params")

        # Create executor for calculating style representation
        net = new(ctx, node, nothing, arg_map, grad_map, style_layers, [],
            fill(mx.zeros(0), num_style), content_layers, [],
            fill(mx.zeros(0), num_content))
        reset_executor!(net)

        # Get Gramian matrices for style image
        net.arg_map[:data][:] = style_arr
        mx.forward(net.exec)
        num_style = size(style_layers, 1)
        for i = 1:num_style
            push!(net.style_repr, net.exec.outputs[num_content + i])
        end

        # Fit network outputs for the content image
        net.arg_map[:data] =
            mx.copy(reshape(content_arr, content_data_size), net.ctx)
        net.grad_map[:data] = mx.zeros(content_data_size, net.ctx)
        reset_executor!(net)

        # Get ReLU output for content image
        mx.forward(net.exec)
        for i = 1:num_content
            push!(net.content_repr, net.exec.outputs[i])
        end

        return net
    end
end

function load_arguments!(
    arg_names :: Array{Symbol,1}
    arg_shapes :: Array{Tuple,1}
    arg_map :: Dict{Symbol, mx.NDArray},
    grad_map :: Dict{Symbol, mx.NDArray},
    model_path :: AbstractString)

    # Model is pre-trained. Get map from symbol to NDArray of weights or biases.
    pretrain = mx.load(model_path, mx.NDArray)

    # Zero-initialize arguments and gradients
    grad_map = Dict(zip(arg_names, [mx.zeros(shape) for shape in arg_shapes]))
    arg_map = Dict(zip(arg_names, [mx.zeros(shape) for shape in arg_shapes]))

    # Copy pre-trained weights and biases into new NDArrays
    for name in arg_names[2:end] # skip :data
        arg_map[name] = pretrain[symbol("arg:" * string(name))]
    end
    gc() # Clean up unused zero NDArrays
end

function reset_executor!(net :: StyleNet)
    if net.exec != nothing
        # Free GPU memory
        net.exec.outputs = []
        gc()
    end

    # Create new executor with different input/output shapes
    net.exec = mx.bind(net.node, net.ctx, net.arg_map, args_grad=net.grad_map)
end
