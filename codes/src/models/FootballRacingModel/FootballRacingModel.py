from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, LeakyReLU, Concatenate
from tensorflow.keras.optimizers import Adam
from codes.utils.models.model_utils import set_seed
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import CategoricalCrossentropy
from codes.src.models.VAE.VAE import VAEClass
from codes.src.models.LSTMVAE.LSTMVAE import LSTMVAEClass


class FootballRacingModelClass:
    def __init__(self, vae_config, lstm_vae_config, racing_config):
        self.vae_config = vae_config
        self.lstm_vae_config = lstm_vae_config
        self.racing_config = racing_config
        self.vae = VAEClass(vae_config)
        self.lstm_vae = LSTMVAEClass(lstm_vae_config)
        self.model = self.get_model()

    def freeze_encoders(self):
        for layer in self.vae.encoder.layers:
            layer.trainable = False
        for layer in self.lstm_vae.encoder.layers:
            layer.trainable = False
        self.model.layers[2].trainable = False
        self.model.layers[3].trainable = False
        return self

    def get_model(self):
        set_seed(self.racing_config.seed)
        input_x1 = Input(self.vae_config.input_dim, name="x1_input")
        input_x2 = Input(self.lstm_vae_config.input_dim, name="x2_input")
        vae_output = self.vae.encoder(input_x1)
        lstm_vae_output = self.lstm_vae.encoder(input_x2)
        concatted_output = Concatenate()([vae_output, lstm_vae_output])
        nn_layer_1 = Dense(self.racing_config.layer_1_dim)(concatted_output)
        nn_layer_1 = BatchNormalization()(nn_layer_1)
        nn_layer_1 = LeakyReLU()(nn_layer_1)
        nn_layer_2 = Dense(self.racing_config.layer_2_dim)(nn_layer_1)
        nn_layer_2 = BatchNormalization()(nn_layer_2)
        nn_layer_2 = LeakyReLU()(nn_layer_2)
        nn_layer_3 = Dense(self.racing_config.layer_3_dim)(nn_layer_2)
        nn_layer_3 = BatchNormalization()(nn_layer_3)
        nn_layer_3 = LeakyReLU()(nn_layer_3)
        nn_layer_4 = Dense(self.racing_config.layer_4_dim)(nn_layer_3)
        nn_layer_4 = BatchNormalization()(nn_layer_4)
        nn_layer_4 = LeakyReLU()(nn_layer_4)
        outputs = Dense(self.racing_config.target_dim, activation='softmax')(nn_layer_4)
        racing_model = Model([input_x1, input_x2], outputs)
        # racing_model.compile(loss=CategoricalCrossentropy(), optimizer=Adam(lr=self.racing_config.learning_rate))
        return racing_model

    def train_model(self, data_x1, data_x2, data_y, validation_data):
        self.vae.train_model(data_x1, data_x1, (validation_data[0][0], validation_data[0][0]))
        self.lstm_vae.train_model(data_x2, data_x2, (validation_data[0][1], validation_data[0][1]))

        self.freeze_encoders()
        print("Freezed layers")

        es = EarlyStopping(patience=self.racing_config.patience, verbose=1, min_delta=self.racing_config.min_delta,
                           monitor='val_loss', mode='auto', restore_best_weights=True)
        # Compile again after freezing the layers
        self.model.compile(loss=CategoricalCrossentropy(), optimizer=Adam(lr=self.racing_config.learning_rate))
        self.model.fit([data_x1, data_x2], data_y,
                       validation_data=validation_data,
                       batch_size=self.racing_config.batch_size,
                       epochs=self.racing_config.epochs,
                       callbacks=[es])
        return self
