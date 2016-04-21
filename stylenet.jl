using MXNet

type Gramian

end

type L2Loss

end

type TVLoss

end

type StyleNet
	exec :: mx.Executor
	data_grad :: mx.NDArray
	
end
