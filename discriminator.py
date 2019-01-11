import tensorflow as tf
import tensorflow.contrib.layers as tcl

class Discriminator(object):
    def __init__(self):
        self.name = 'D_conv'

    def __call__(self, x, local_x, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            global_output = self.global_discriminator(x)
            local_output = self.local_discriminator(local_x)
            with tf.variable_scope('concatenation'):
                output = tf.concat((global_output, local_output), 1)
                output = tcl.fully_connected(output, 2, activation_fn=None)
        return output

    def global_discriminator(self, x):
        size = 96
        shared = tcl.conv2d(x, num_outputs=size, kernel_size=4,  # bzx64x64x3 -> bzx32x32x64
                            stride=2, activation_fn=lrelu)
        shared = tcl.conv2d(shared, num_outputs=size * 2, kernel_size=4,  # 16x16x128
                            stride=2, activation_fn=lrelu, normalizer_fn=tcl.batch_norm)
        shared = tcl.conv2d(shared, num_outputs=size * 4, kernel_size=4,  # 8x8x256
                            stride=2, activation_fn=lrelu, normalizer_fn=tcl.batch_norm)
        shared = tcl.conv2d(shared, num_outputs=size * 8, kernel_size=4,  # 4x4x512
                            stride=2, activation_fn=lrelu, normalizer_fn=tcl.batch_norm)
        shared = tcl.conv2d(shared, num_outputs=size * 16, kernel_size=4,  # 2x2x1024
                            stride=2, activation_fn=lrelu, normalizer_fn=tcl.batch_norm)

        shared = tcl.flatten(shared)
        return shared

    def local_discriminator(self, x):
        #         with tf.variable_scope('local'):
        size = 32
        shared = tcl.conv2d(x, num_outputs=size * 2, kernel_size=4,  # 16x16x128
                            stride=2, activation_fn=lrelu, normalizer_fn=tcl.batch_norm)
        shared = tcl.conv2d(shared, num_outputs=size * 4, kernel_size=4,  # 8x8x256
                            stride=2, activation_fn=lrelu, normalizer_fn=tcl.batch_norm)
        shared = tcl.conv2d(shared, num_outputs=size * 8, kernel_size=4,  # 4x4x512
                            stride=2, activation_fn=lrelu, normalizer_fn=tcl.batch_norm)
        shared = tcl.conv2d(shared, num_outputs=size * 16, kernel_size=4,  # 2x2x1024
                            stride=2, activation_fn=lrelu, normalizer_fn=tcl.batch_norm)

        shared = tcl.flatten(shared)
        return shared

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
