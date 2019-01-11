import tensorflow as tf
import tensorflow.contrib.layers as tcl

class C_conv(object):
    def __init__(self):
        self.name = 'C_conv'

    def __call__(self, x, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            size = 128
            shared = tcl.conv2d(x, num_outputs=size, kernel_size=4, # bzx64x64x3 -> bzx32x32x64
                        stride=2, activation_fn=lrelu)
            shared = tcl.conv2d(shared, num_outputs=size * 2, kernel_size=4, # 16x16x128
                        stride=2, activation_fn=lrelu, normalizer_fn=tcl.batch_norm)
            shared = tcl.conv2d(shared, num_outputs=size * 4, kernel_size=4, # 8x8x256
                        stride=2, activation_fn=lrelu, normalizer_fn=tcl.batch_norm)
            #d = tcl.conv2d(d, num_outputs=size * 8, kernel_size=3, # 4x4x512
            #			stride=2, activation_fn=lrelu, normalizer_fn=tcl.batch_norm)

            shared = tcl.fully_connected(tcl.flatten( # reshape, 1
                        shared), 1024, activation_fn=lrelu, normalizer_fn=tcl.batch_norm)

            q = tcl.fully_connected(tcl.flatten(shared), 128, activation_fn=lrelu, normalizer_fn=tcl.batch_norm)
            q = tcl.fully_connected(q, 300, activation_fn=None) # 10 classes

            return q
    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]