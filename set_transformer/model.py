import tensorflow as tf
from set_transformer.layers.blocks import STEncoderBasic, STDecoderBasic


class SetTransformer(tf.keras.Model):
    def __init__(self, encoder_d=4, m=3, encoder_h=2, out_dim=1, decoder_d=4, decoder_h=2, k=2):
        super(SetTransformer, self).__init__()
        self.basic_encoder = STEncoderBasic(d=encoder_d, m=m, h=encoder_h)
        self.basic_decoder = STDecoderBasic(out_dim=out_dim, d=decoder_d, h=decoder_h, k=k)

    def call(self, x):
        enc_output = self.basic_encoder(x)  # (batch_size, set_len, d_model)
        return self.basic_decoder(enc_output)
