from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Lambda, BatchNormalization, LeakyReLU, LSTM, RepeatVector, TimeDistributed
from tensorflow.keras.optimizers import Adam
from codes.utils.models.model_utils import sampling, set_seed, vae_loss


class LSTMVAEClass:
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
        encoder_input = Input(self.config.input_dim)
        self.encoder_input_layer = encoder_input
        lstm_layer = LSTM(self.config.lstm_dim, return_sequences=False)(encoder_input)
        lstm_layer = BatchNormalization()(lstm_layer)
        lstm_layer = LeakyReLU()(lstm_layer)
        encoder_layer1 = Dense(self.config.layer_1_dim)(lstm_layer)
        encoder_layer1 = BatchNormalization()(encoder_layer1)
        encoder_layer1 = LeakyReLU()(encoder_layer1)
        encoder_mu = Dense(self.config.latent_dim)(encoder_layer1)
        encoder_log_variance = Dense(self.config.latent_dim)(encoder_layer1)
        encoder_output = Lambda(sampling)([encoder_mu, encoder_log_variance])
        self.encoder_mu = encoder_mu
        self.encoder_log_variance = encoder_log_variance
        encoder = Model(encoder_input, encoder_output)
        return encoder

    def get_decoder(self):
        set_seed(self.config.seed)
        decoder_input = Input(self.config.latent_dim)
        repeat_vector = RepeatVector(self.config.input_dim[0])(decoder_input)
        lstm_decoder1 = LSTM(self.config.lstm_dim, return_sequences=True)(repeat_vector)
        lstm_decoder1 = BatchNormalization()(lstm_decoder1)
        lstm_decoder1 = LeakyReLU()(lstm_decoder1)
        decoder_layer1 = Dense(self.config.layer_1_dim)(lstm_decoder1)
        decoder_layer1 = BatchNormalization()(decoder_layer1)
        decoder_layer1 = LeakyReLU()(decoder_layer1)
        decoder_output = TimeDistributed(Dense(self.config.input_dim[1]))(decoder_layer1)
        self.decoder_output_layer = decoder_output
        decoder = Model(decoder_input, decoder_output)
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
