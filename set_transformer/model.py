import tensorflow as tf
from set_transformer.layers.blocks import STEncoderBasic, STDecoderBasic


class SetTransformer(tf.keras.Model):
    def __init__(self, ):
        super(SetTransformer, self).__init__()
        self.basic_encoder = STEncoderBasic(d=4, m=3, h=2)
        self.basic_decoder = STDecoderBasic(out_dim=1, d=4, m=2, h=2, k=2)

    def call(self, x):
        enc_output = self.basic_encoder(x)  # (batch_size, set_len, d_model)
        return self.basic_decoder(enc_output)
