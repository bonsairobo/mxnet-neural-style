using MXNet

type ExecutorData
    node :: mx.SymbolicNode
    exec :: mx.Executor
    data :: mx.NDArray
    data_grad :: mx.NDArray
end
