# spherenet

This is an implementation of [Deep Hyperspherical Learning](https://arxiv.org/abs/1711.03189v1). Deep Hyperspherical Learning is a simple modification to the convolution operator that speeds up convergence and improves accuracy on some tasks.

# Usage

Install the `spherenet` package:

```
pip install spherenet
```

## SphereConv layers

To use a SphereConv layer, do something like this:

```
from spherenet import sphere_conv

...

output = sphere_conv(input_image, NUM_FILTERS, KERNEL_SIZE, strides=STRIDE,
                     variant='sigmoid', sigmoid_k=0.3)
```

The variant argument can be `linear`, `cosine`, or `sigmoid`. You can also pass a `regularization` argument to specify a regularization coefficient, in which case a regularization term is added to `tf.GraphKeys.REGULARIZATION_LOSSES`.

In general, the `sphere_conv` function is similar to [tf.layers.conv2d](https://www.tensorflow.org/api_docs/python/tf/layers/conv2d). For complete usage of `sphere_conv`, run:

```
python -c 'import spherenet; help(spherenet.sphere_conv)'
```
