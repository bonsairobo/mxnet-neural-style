type ExecutorData
    nodes :: Dict{Symbol, mx.SymbolicVariable}
    exec :: mx.Executor
    data :: mx.NDArray
    data_grad :: mx.NDArray
end

function forward_arr(exec_dat :: ExecutorData, input)
    exec_dat.data[:] = input
    mx.forward(exec_dat.exec)
    return exec_dat.exec.outputs
end

include("vggnet.jl")
include("gramian.jl")
