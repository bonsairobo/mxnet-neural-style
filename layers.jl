using MXNet

function make_gramian(layer, in_shape)
    # Flatten the activation for each filter to form a matrix where each column
    # is one filter's activation
    flat_shape = (reduce(*, in_shape[1:end-1]), in_shape[end])

    # Multiply activation matrix by its transpose to make Gramian matrix
    reshape = mx.Reshape(data=layer, target_shape=flat_shape)
    tpose = mx.SwapAxis(data=reshape, dim1=1, dim2=2)
    gram = mx.FullyConnected(data=reshape, weight=tpose, no_bias=true,
        num_hidden=in_shape[end])

    return gram
end

function l2_gradient(nd1 :: mx.NDArray, nd2 :: mx.NDArray)
    
end

function tv_gradient(img_nd :: mx.NDArray)

end
