using MXNet

type Gramian
	data :: mx.NDArray
	data_grad :: mx.NDArray
	style_repr_nd :: mx.NDArray
end

function get_style_repr(style_img_nd)

end

function make_gramian(in_shape)
	# Each entry of the Gramian matrix is the correllation between activations
    # of two filters in a layer
    style_relu1 = mx.Variable(:style_relu1)
    style_relu2 = mx.Variable(:style_relu2)

    # Flatten the activation for each filter to form a matrix where each column
    # is one filter's activation
    flat_shape = (reduce(*, in_shape[1:end-1]), in_shape[end])
    flat_act1 = mx.Reshape(data=style_relu1, target_shape=flat_shape)
    flat_act2 = mx.Reshape(data=style_relu2, target_shape=flat_shape)

    # Dot all 2-combinations of activations to create Gramian. This is
    # equivalent to matrix self-multiplication with transpose or a MLP where the
    # input and weights are each others transpose.
    flat_act2 = mx.SwapAxis(data=flat_act2, dim1=)
    fc = mx.FullyConnected(data=flat_act1, weight=flat_act2, no_bias=true,
        num_hidden=in_shape[end])
        
    return Gramian()
end

type L2Norm
	data :: mx.NDArray
	data_grad :: mx.NDArray
end

type TotalVariation
	data :: mx.NDArray
	data_grad :: mx.NDArray
end

type StyleNet
	exec :: mx.Executor
	gram :: Gramian
	dist :: L2Norm
	total_var :: TotalVariation
end

function make_stylenet()
	# Group loss layers for convenient access to all unbound arguments
    # (data, weights, biases) on which the loss is dependent
    style_group = mx.Group(style_layers...)
    content_group = mx.Group(content_layer...)
    loss_group = mx.Group(style_group, content_group)
    
	get_style_repr()
	
	return StyleNet()
end
