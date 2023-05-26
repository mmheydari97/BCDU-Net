import tensorflow as tf

class TimeAwareConvLSTMCell(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, padding, activation, frame_size):
        super(TimeAwareConvLSTMCell, self).__init__()
        
        if activation == "tanh":
            self.activation = tf.nn.tanh
        elif activation == "relu":
            self.activation = tf.nn.relu
        
        self.conv = tf.keras.layers.Conv2D(
            filters=4 * out_channels,
            kernel_size=kernel_size,
            padding=padding)
        
        self.W_ci = self.add_weight(
            shape=(out_channels, *frame_size),
            initializer=tf.keras.initializers.GlorotUniform(),
            trainable=True)
        
        self.W_co = self.add_weight(
            shape=(out_channels, *frame_size),
            initializer=tf.keras.initializers.GlorotUniform(),
            trainable=True)
        
        self.W_cf = self.add_weight(
            shape=(out_channels, *frame_size),
            initializer=tf.keras.initializers.GlorotUniform(),
            trainable=True)
        
        self.decay_factor = self.add_weight(
            shape=(1,),
            initializer=tf.keras.initializers.GlorotUniform(),
            trainable=True)

    def call(self, X, H_prev, C_prev, duration):
        conv_output = self.conv(tf.concat([X, H_prev], axis=3))
        i_conv, f_conv, C_conv, o_conv = tf.split(conv_output, num_or_size_splits=4, axis=3)

        input_gate = tf.sigmoid(i_conv + tf.expand_dims(self.W_ci, axis=0) * C_prev)
        decay_factor = tf.exp(-self.decay_factor * duration)
        forget_gate = tf.sigmoid(f_conv + tf.expand_dims(self.W_cf, axis=0) * C_prev) * decay_factor
        C = forget_gate * C_prev + input_gate * self.activation(C_conv)
        output_gate = tf.sigmoid(o_conv + tf.expand_dims(self.W_co, axis=0) * C)
        H = output_gate * self.activation(C)

        return H, C
