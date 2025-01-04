import tensorflow as tf
import numpy as np

@tf.keras.utils.register_keras_serializable()
class InferenceDropout(tf.keras.layers.Dropout):
    def __init__(self, rate, **kwargs):
        super().__init__(rate, **kwargs)

    def call(self, inputs, training=None):
        return super().call(inputs, training=True)

    def get_config(self):
        # config = super().get_config()
        return super().get_config()

# @tf.keras.utils.register_keras_serializable()
# class MCDropout(tf.keras.layers.Layer):
#     def __init__(self, layer, samples=10, **kwargs):
#         super(MCDropout, self).__init__(**kwargs)
#         self.layer = layer
#         self.samples = samples

#     def call(self, inputs, training=None):
#         outputs = [self.layer(inputs, training=True) for _ in range(self.samples)]
#         # print(outputs)
#         mean = tf.reduce_mean(outputs, axis=0)
#         stdev = tf.math.reduce_std(outputs, axis=0)
#         return tf.concat([mean, stdev], axis=-1)

#     def get_config(self):
#         config = super(MCDropout, self).get_config()
#         config.update({
#             "layer": tf.keras.layers.serialize(self.layer),
#             "samples": self.samples
#         })
#         return config

#     @classmethod
#     def from_config(cls, config):
#         layer = tf.keras.layers.deserialize(config.pop('layer'))
#         sample = config.pop('samples')
#         return cls(layer, sample, **config)

#     def compute_output_shape(self, input_shape):
#         return (input_shape[0], 2)

# def gaussian_loss(y_true, y_pred):
#     mean = y_pred[:, 0]
#     stdev = tf.math.maximum(y_pred[:, 1], 1e-6)

#     stdev2 = tf.math.log(tf.math.pow(stdev, 2))
#     z = tf.math.multiply(2*np.pi, stdev2)
#     z = tf.math.pow(z, 0.5)

#     arg = -0.5*(y_true-mean)
#     arg = tf.math.pow(arg, 2)
#     arg = arg / stdev2)

#     return tf.reduce_mean(tf.divide(tf.exp(arg), z))
    # print(mean.numpy(), stdev.numpy())
    
    # likelihood = -0.5 * tf.math.log(2.0 * tf.constant(3.141592653589793, dtype=tf.float32)) \
    #              - tf.math.log(stdev) \
    #              - 0.5 * tf.math.square((y_true - mean) / stdev)
    # return -tf.reduce_mean(likelihood)