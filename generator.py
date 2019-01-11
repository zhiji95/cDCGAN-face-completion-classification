import tensorflow as tf
import tensorflow.contrib.layers as tcl

class Generator(object):
    def __init__(self):
        self.name = 'G_dia'

    def __call__(self, x):
        with tf.variable_scope(self.name) as scope:
            g = self.downsample(x)
            with tf.variable_scope('dilated1'):
                g = self.dilated_conv_layer(g, [3, 3, 1024, 1024], 2)
                g = self.lrelu(tf.layers.batch_normalization(g))
                print("dilated layer 1", g.get_shape().as_list())
            with tf.variable_scope('dilated2'):
                g = self.dilated_conv_layer(g, [3, 3, 1024, 1024], 4)
                g = self.lrelu(tf.layers.batch_normalization(g))
                print("dilated layer 2", g.get_shape().as_list())
            with tf.variable_scope('dilated3'):
                g = self.dilated_conv_layer(g, [3, 3, 1024, 1024], 8)
                g = self.lrelu(tf.layers.batch_normalization(g))
                print("dilated layer 3", g.get_shape().as_list())
            with tf.variable_scope('dilated4'):
                g = self.dilated_conv_layer(g, [3, 3, 1024, 1024], 16)
                g = self.lrelu(tf.layers.batch_normalization(g))
                print("dilated layer 4", g.get_shape().as_list())

            img = self.build_up_resnet(g)
            g = tf.nn.sigmoid(img)

            return g

    def build_down_resnet(self, x):
        conv1 = tf.layers.conv2d(x, 64, (3, 3), padding='same',
                                 kernel_initializer=tcl.xavier_initializer(), name='conv1')
        with tf.variable_scope('block1'):
            block1 = self.build_residual_block(conv1, 64, (2, 2))
            print("residual block 1", block1.get_shape().as_list())
        with tf.variable_scope('block2'):
            block2 = self.build_residual_block(block1, 128, (2, 2))
            print("residual block 2", block2.get_shape().as_list())
        with tf.variable_scope('block3'):
            block3 = self.build_residual_block(block2, 256, (2, 2))
        return block3

    def dilated_conv_layer(self, x, filter_shape, dilation):
        filters = tf.get_variable(
            name='weight',
            shape=filter_shape,
            dtype=tf.float32,
            initializer=tf.contrib.layers.xavier_initializer(),
            trainable=True)
        return tf.nn.atrous_conv2d(x, filters, dilation, padding='SAME')

    def batch_normalize(x, is_training=True, decay=0.99, epsilon=0.001):
        def bn_train():
            batch_mean, batch_var = tf.nn.moments(x, axes=[0, 1, 2])
            train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
            train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
            with tf.control_dependencies([train_mean, train_var]):
                return tf.nn.batch_normalization(x, batch_mean, batch_var, beta, scale, epsilon)

        def bn_inference():
            return tf.nn.batch_normalization(x, pop_mean, pop_var, beta, scale, epsilon)

        dim = x.get_shape().as_list()[-1]
        beta = tf.get_variable(
            name='beta',
            shape=[dim],
            dtype=tf.float32,
            initializer=tf.truncated_normal_initializer(stddev=0.0),
            trainable=True)
        scale = tf.get_variable(
            name='scale',
            shape=[dim],
            dtype=tf.float32,
            initializer=tf.truncated_normal_initializer(stddev=0.1),
            trainable=True)
        pop_mean = tf.get_variable(
            name='pop_mean',
            shape=[dim],
            dtype=tf.float32,
            initializer=tf.constant_initializer(0.0),
            trainable=False)
        pop_var = tf.get_variable(
            name='pop_var',
            shape=[dim],
            dtype=tf.float32,
            initializer=tf.constant_initializer(1.0),
            trainable=False)

        return tf.cond(is_training, bn_train, bn_inference)

    def downsample(self, x):
        with tf.variable_scope('Downsample'):
            feat = self.build_down_resnet(x)
            feat = tf.layers.conv2d(feat, 1024, (1, 1), padding='same', strides=(1, 1),
                                    kernel_initializer=tcl.xavier_initializer())

            return feat

    def build_up_resnet(self, feat_reshape):

        with tf.variable_scope('block3'):
            block3 = self.build_residual_block(feat_reshape, 256, (2, 2), transpose=True)
            print("buildup block 3", block3.get_shape().as_list())
        with tf.variable_scope('block4'):
            block4 = self.build_residual_block(block3, 128, (2, 2), transpose=True)
            print("buildup block 4", block4.get_shape().as_list())
        with tf.variable_scope('block5'):
            block5 = self.build_residual_block(block4, 64, (2, 2), transpose=True)
            print("buildup block 5", block5.get_shape().as_list())
        deconv = tf.layers.conv2d_transpose(block5, 32, (4, 4), padding='same',
                                             kernel_initializer=tcl.xavier_initializer(), name='deconv')
        out = tf.layers.conv2d(deconv, 3, (3, 3), padding='same',
                               kernel_initializer=tcl.xavier_initializer(), name='output')

        return out

    def build_residual_block(self, input_, channel, strides, transpose=False):
        if not transpose:
            bn = self.lrelu(tf.layers.batch_normalization(input_))
            conv1 = tf.layers.conv2d(bn, channel, (3, 3), padding='same', strides=strides,
                                     kernel_initializer=tcl.xavier_initializer())
            conv2 = self.lrelu(tf.layers.batch_normalization(conv1))
            conv2 = tf.layers.conv2d(conv2, channel, (3, 3), padding='same',
                                     kernel_initializer=tcl.xavier_initializer())
            conv3 = tf.layers.conv2d(input_, channel, (1, 1), strides=strides,
                                     kernel_initializer=tcl.xavier_initializer())
            out = tf.add(conv3, conv2)
        else:
            bn = tf.nn.relu(tf.layers.batch_normalization(input_))
            deconv1 = tf.layers.conv2d_transpose(bn, channel, (3, 3), padding='same', strides=strides,
                                                 kernel_initializer=tcl.xavier_initializer())
            deconv2 = tf.nn.relu(tf.layers.batch_normalization(deconv1))
            deconv2 = tf.layers.conv2d(deconv2, channel, (3, 3), padding='same',
                                                 kernel_initializer=tcl.xavier_initializer())
            deconv3 = tf.layers.conv2d(input_, channel, (1, 1), strides=strides,
                                                 kernel_initializer=tcl.xavier_initializer())
            out = tf.add(deconv3, deconv2)

        return out

    def lrelu(self, x, leak=0.2):
        return tf.maximum(x, leak * x)

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)


class D_conv(object):
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


