using MXNet

include("executors.jl")
include("gramian.jl")
include("vggnet.jl")

type StyleNet
    ctx :: mx.Context

    # Computational graph (forward/backward) executors
    vgg_exec :: ExecutorData
    gram_execs :: Array{ExecutorData,1}

    style_repr :: Array{mx.NDArray,1} # Gramian matrices
    content_repr :: Array{mx.NDArray,1} # ReLU outputs

    content_grads :: Array{mx.NDArray,1} # L2 gradient

    # Total Variation gradient for image
    tv_grad :: mx.NDArray

    out_shapes :: Array{Tuple,1}

    num_content :: Int64
    num_style :: Int64

    function StyleNet(ctx, content_img, style_img, content_layers, style_layers)
        num_content = size(content_layers, 1)
        num_style = size(style_layers, 1)

        content_arr = preprocess_vgg(content_img)
        content_size = size(content_arr)

        # Make executors
        vgg_exec, out_shapes = make_vgg_executor(
            vcat(content_layers, style_layers), content_size, ctx)
        gram_execs =
            [make_gramian_executor(shape, ctx)
             for shape in out_shapes[num_content+1:end]]

        # Allocate GPU memory for representation gradients
        content_grads =
            map(x -> mx.zeros(size(x), ctx),
                vgg_exec.exec.outputs[1:num_content])

        # Get Gramian matrices for style image
        vgg_exec.data[:] = preprocess_vgg(style_img)
        mx.forward(vgg_exec.exec)
        for (i, gram) in enumerate(gram_execs)
            gram.data[:] = vgg_exec.exec.outputs[num_content+i]
            mx.forward(gram.exec)
        end
        style_repr = map(x -> mx.copy(x.exec.outputs[1], ctx), gram_execs)
    
        # Get ReLU output for content image
        vgg_exec.data[:] = preprocess_vgg(content_img)
        mx.forward(vgg_exec.exec)
        content_repr =
            map(x -> mx.copy(x, ctx), vgg_exec.exec.outputs[1:num_content])

        # Initialize data to noise for optimization
        #vgg_exec.data[:] = mx.rand(-0.1, 0.1, content_size)
        vgg_exec.data[:] = mx.ones(content_size, ctx)

        return new(ctx, vgg_exec, gram_execs, style_repr,
            content_repr, content_grads, mx.zeros(0), out_shapes,
            num_content, num_style)
    end
end

function optimize(net :: StyleNet)
    lr = mx.LearningRate.Fixed(0.1)
    sgd = mx.SGD(
        lr = 0.1,
        momentum = 0.9,
        weight_decay = 0.005,
        lr_scheduler = lr,
        grad_clip = 10)
    sgd_state = mx.create_state(sgd, 0, net.vgg_exec.data)
    sgd.state = mx.OptimizationState(10)

    for epoch = 1:500
        println(epoch)

        mx.forward(net.vgg_exec.exec)

        # Calculate style gradients
        for (i, gram) in enumerate(net.gram_execs)
            copy!(gram.data, net.vgg_exec.exec.outputs[net.num_content+i])
            mx.forward(gram.exec)
            gram_grad = gram.exec.outputs[1] - net.style_repr[i]
            mx.backward(gram.exec, [gram_grad])
            mx.div_from!(
                gram.data_grad,
                (net.out_shapes[i][3] ^ 2) * reduce(*, net.out_shapes[i][1:2]))
        end

        # Calculate content gradients
        for (i, relu_out) in
        enumerate(net.vgg_exec.exec.outputs[1:net.num_content])
            net.content_grads[i][:] = relu_out - net.content_repr[i]
        end

        # VGG backprop
        grads = vcat(net.content_grads, map(x -> x.data_grad, net.gram_execs))
        mx.backward(net.vgg_exec.exec, grads)

        # Update image
        mx.update(sgd, 0, net.vgg_exec.data, net.vgg_exec.data_grad, sgd_state)
    end

    # Convert NDArray into Image
    out_arr = net.vgg_exec.data |> size |> zeros
    copy!(out_arr, net.vgg_exec.data)
    return postprocess_vgg(out_arr)
end
