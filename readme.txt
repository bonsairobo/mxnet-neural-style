3rd-Party Software:
- (Ubuntu 15.10)
- julia 0.4
- MXNet.jl
- Images.jl
- DocOpt.jl
- libmxnet.so
- libCUDA
- libCUDNN

Installing the Julia dependencies with Pkg.add() is sufficient, though you
may need to set the environment variable MXNET_HOME for MXNet.jl if you compile
your own libmxnet.so.

I've included libmxnet.so compiled on Ubuntu 15.10 using clang. It also links
with CUDA. If you need to build your own version, I've included my config.mk
that I used in the process. If you want to use your CPU, just change 'mx.GPU()'
to 'mx.CPU()' in 'imgstyle.jl', but be warned, it will be slow.

How to run:

./imgstyle.jl -h

E.g.

./imgstyle.jl input/the_scream.jpg input/trippy.jpg --content_weight 20 --style_weight 1 --long_edge 800 --num_iter 700C

You WILL need to toy with the style/content weights to get a good result.

If you get a CUDA memory error, use the --long_edge option to scale down the
content image.
