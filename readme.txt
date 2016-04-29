3rd-Party Software:
- (Ubuntu 15.10)
- julia 0.4
- MXNet.jl
- Images.jl
- DocOpt.jl
- libmxnet.so
- libCUDA
- libCUDNN
- imagemagick

Installing the Julia dependencies with Pkg.add() is sufficient, though you
may need to set the environment variable MXNET_HOME for MXNet.jl if you compile
your own libmxnet.so.

You can download my copy of libmxnet.so from Dropbox
(https://www.dropbox.com/sh/tm08a1umx4eyl2n/AABBQzaQFA6Ui26GX9gcDdHra?dl=0)
compiled on Ubuntu 15.10 using clang. It also links with CUDA. If you need to
build your own version, I've included my config.mk that I used in the process.
If you want to use your CPU, just change 'mx.GPU()' to 'mx.CPU()' in
'imgstyle.jl', but be warned, it will be slow.

How to run:

./wget_model.sh # Download the pretrained VGG19 network

./imgstyle.jl -h

E.g.

./imgstyle.jl input/the_scream.jpg input/trippy.jpg --content_weight 20 --style_weight 1 --long_edge 800 --num_iter 700

You WILL need to toy with the style/content weights to get a good result.

If you get a CUDA out of memory error, use the --long_edge option to scale down
the content image.
