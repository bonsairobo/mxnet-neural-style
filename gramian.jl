using MXNet

include("executors.jl")

function make_gramian_executor(in_shape, ctx)
    data = mx.Variable(:gram_data)
    weight = mx.Variable(:gram_weight)

    # Flatten the activation for each filter to form a matrix where each
    # column is one filter's activation
    flat_shape = (reduce(*, in_shape[1:2]), in_shape[3])
    flat_data = mx.Reshape(data=data, target_shape=flat_shape)
    flat_weight = mx.Reshape(data=weight, target_shape=flat_shape)

    # Multiply activation matrix by its transpose to make Gramian matrix
    node = mx.FullyConnected(data=flat_data, weight=flat_weight, no_bias=true,
        num_hidden=in_shape[3])

    # Allocate GPU memory for input to Gramian
    data_nd = mx.zeros(in_shape, ctx)
    grad_nd = mx.zeros(in_shape, ctx)

    # Make executor
    args = Dict(:gram_data => data_nd, :gram_weight => data_nd)
    grad = Dict(:gram_data => grad_nd)
    reqs = Dict(:gram_data => mx.GRAD_WRITE, :gram_weight => mx.GRAD_NOP)
    exec = mx.bind(node, ctx, args, args_grad=grad, grad_req=reqs)

    return ExecutorData(node, exec, data_nd, grad_nd)
end
