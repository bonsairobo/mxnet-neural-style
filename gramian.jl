using MXNet

function make_gramian(layer, in_shape)
    # Flatten the activation for each filter to form a matrix where each column
    # is one filter's activation
    flat_shape = (reduce(*, in_shape[1:2]), in_shape[3])

    # Multiply activation matrix by its transpose to make Gramian matrix
    reshape = mx.Reshape(data=layer, target_shape=flat_shape)
    gram = mx.FullyConnected(data=reshape, weight=reshape, no_bias=true,
        num_hidden=flat_shape[2])

    return gram
end
