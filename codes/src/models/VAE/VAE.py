from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Lambda, BatchNormalization, LeakyReLU
from tensorflow.keras.optimizers import Adam
from codes.utils.models.model_utils import sampling, set_seed, vae_loss


class VAEClass:
    def __init__(self, config):
        self.config = config
        self.encoder_input_layer = None
        self.decoder_output_layer = None
        self.encoder_mu = None
        self.encoder_log_variance = None
        self.encoder = self.get_encoder()
        self.decoder = self.get_decoder()
        self.model = self.get_model()

    def get_encoder(self):
        set_seed(self.config.seed)
        encoder_input = Input(self.config.input_dim, name="original_input")
        self.encoder_input_layer = encoder_input
        encoder_layer1 = Dense(self.config.layer_1_dim)(encoder_input)
        encoder_layer1 = BatchNormalization(name="encoder_norm_1")(encoder_layer1)
        encoder_layer1 = LeakyReLU(name="encoder_actv_layer_1")(encoder_layer1)
        encoder_layer2 = Dense(self.config.layer_2_dim)(encoder_layer1)
        encoder_layer2 = BatchNormalization(name="encoder_norm_2")(encoder_layer2)
        encoder_layer2 = LeakyReLU(name="encoder_actv_layer_2")(encoder_layer2)
        encoder_mu = Dense(self.config.latent_dim)(encoder_layer2)
        encoder_log_variance = Dense(self.config.latent_dim)(encoder_layer2)
        encoder_output = Lambda(sampling, name="encoder_output")([encoder_mu,
                                                                  encoder_log_variance])
        self.encoder_mu = encoder_mu
        self.encoder_log_variance = encoder_log_variance
        encoder = Model(encoder_input, encoder_output, name='encoder_model')
        return encoder

    def get_decoder(self):
        set_seed(self.config.seed)
        decoder_input = Input(self.config.latent_dim, name="decoder_input")
        decoder_layer1 = Dense(self.config.layer_1_dim)(decoder_input)
        decoder_layer1 = BatchNormalization(name="decoder_norm_1")(decoder_layer1)
        decoder_layer1 = LeakyReLU(name="decoder_actv_layer_1")(decoder_layer1)
        decoder_layer2 = Dense(self.config.layer_2_dim)(decoder_layer1)
        decoder_layer2 = BatchNormalization(name="decoder_norm_2")(decoder_layer2)
        decoder_layer2 = LeakyReLU(name="decoder_actv_layer_2")(decoder_layer2)
        decoder_output = Dense(self.config.input_dim, name="decoder_output")(decoder_layer2)
        self.decoder_output_layer = decoder_output
        decoder = Model(decoder_input, decoder_output, name='decoder_model')
        return decoder

    def get_model(self):
        encoder_output = self.encoder(self.encoder_input_layer)
        decoder_output = self.decoder(encoder_output)
        vae = Model(self.encoder_input_layer, decoder_output)
        vae.add_loss(vae_loss(self.encoder_input_layer, decoder_output,
                              self.encoder_log_variance, self.encoder_mu))
        vae.compile(loss=None, optimizer=Adam(lr=self.config.learning_rate))
        return vae

    def train_model(self, data_x, data_y, validation_data):
        es = EarlyStopping(patience=self.config.patience, verbose=1, min_delta=self.config.min_delta,
                           monitor='val_loss', mode='auto', restore_best_weights=True)
        self.model.fit(data_x, data_y,
                       validation_data=validation_data,
                       batch_size=self.config.batch_size,
                       epochs=self.config.epochs,
                       callbacks=[es])
        return self
