import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
from sklearn.metrics import confusion_matrix
import tensorflow as tf

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

from codes.src.models.FootballRacingModel.FootballRacingModel import FootballRacingModelClass
from codes.config import vae_config, lstm_vae_config, football_racing_config
from codes.utils.data.generate_vae_data import generate_vae_data
from codes.utils.data.generate_lstm_vae_data import generate_lstm_vae_data
from codes.utils.data.generate_target_data import generate_target_data

# Load data
df_dict = generate_vae_data("data")
df_dict = generate_lstm_vae_data("data", df_dict)
df_dict = generate_target_data("data", df_dict)
print(df_dict['train']['x1'].shape, df_dict['train']['x2'].shape, df_dict['train']['y'].shape)
print(df_dict['val']['x1'].shape, df_dict['val']['x2'].shape, df_dict['val']['y'].shape)
print(df_dict['test']['x1'].shape, df_dict['test']['x2'].shape, df_dict['test']['y'].shape)

# Train Model
football_racing = FootballRacingModelClass(vae_config, lstm_vae_config, football_racing_config)
football_racing.freeze_encoders()

es = EarlyStopping(patience=football_racing.racing_config.patience, verbose=1, min_delta=football_racing.racing_config.min_delta,
                   monitor='val_loss', mode='auto', restore_best_weights=True)
# Compile again after freezing the layers
football_racing.model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=Adam(lr=football_racing.racing_config.learning_rate))

football_racing.vae.model.load_weights('models/vae.h5')
football_racing.lstm_vae.model.load_weights('models/lstm_vae.h5')
football_racing.model.load_weights('models/football_racing_model.h5')

# Predict test results
data_x1, data_x2, data_y = df_dict['train']['x1'], df_dict['train']['x2'], df_dict['train']['y']
val_x1, val_x2, val_y = df_dict['val']['x1'], df_dict['val']['x2'], df_dict['val']['y']
test_x1, test_x2, test_y = df_dict['test']['x1'], df_dict['test']['x2'], df_dict['test']['y']

# Predict 100 times to get win/draw/lose distribution, select row mode as final prediction value
# Note we can use these 100 prediction results to quantify the prediction uncertainty
pred_list = []
for i in range(100):
    pred_list.append(np.argmax(football_racing.model.predict([test_x1, test_x2]), axis=1))
prediction_df = pd.DataFrame(pred_list).T
final_prediction = prediction_df.mode(axis=1)[0].astype(int)

# Get actual results
actual_game_result = np.argmax(df_dict['test']['y'], axis=1)
actual_game_result = pd.Series(actual_game_result)

# Eval results
# Note: The reproduce results will be different from the documented as random samples are taken from the
# latent space of VAE and LSTMVAE. But the POC of Distribution of Confidence comparison will not change
## Accuracy
print("Accuracy:", round((final_prediction == actual_game_result).sum()/660*100, 2))
# # # Accuracy: 58.48%

## Confusion Matrix results
print(confusion_matrix(actual_game_result, final_prediction))
# array([[187,  23,  43],
#        [ 65,  16,  73],
#        [ 52,  18, 183]], dtype=int64)

# Distribution of confidence comparison
prediction_df[(final_prediction == actual_game_result)].std(axis=1).hist()
# plt shown in 'Results/correct_prediction_prob_std_distribution.png'
prediction_df[(final_prediction != actual_game_result)].std(axis=1).hist()
# plt shown in 'Results/incorrect_prediction_prob_std_distribution.png'

fig = plt.figure(figsize=(10, 6))
seaborn.distplot(prediction_df[(final_prediction == actual_game_result)].std(axis=1), hist=False)
seaborn.distplot(prediction_df[(final_prediction != actual_game_result)].std(axis=1), hist=False)
fig.legend(labels=['Correct_predictions_sample_std','Incorrect_predictions_sample_std'])
plt.title("Sample std of correct vs incorrect prediction")
plt.show()  # plt saved under 'Results/Sample_std_of_correct_vs_incorrect_prediction.png'
