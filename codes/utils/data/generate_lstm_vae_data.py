import pandas as pd
import numpy as np
from codes.utils.data.get_data import get_data


def transform_data(df):
    # performance transformation of calculate the difference between expected stats and actual stats in the games
    df['xGD'] = df['scored'] - df['xG']
    df['xGAD'] = df['conceded'] - df['xGA']
    df['xptsD'] = df['pts'] - df['xpts']
    df['deepD'] = df['deep'] - df['deep_allowed']
    df['ppda_attk_diff'] = df.apply(lambda x: x['ppda']['att'] - x['ppda_allowed']['att'], axis=1)
    df['ppda_def_diff'] = df.apply(lambda x: x['ppda']['def'] - x['ppda_allowed']['def'], axis=1)
    return df


def gen_seq(id_df, seq_length, seq_cols):
    # generate sequence for LSTM input
    data_matrix = id_df[seq_cols]
    num_elements = data_matrix.shape[0]
    for start, stop in zip(range(0, num_elements - seq_length, 1), range(seq_length, num_elements, 1)):
        yield data_matrix[stop - seq_length:stop].values.reshape((-1, len(seq_cols)))


def process_lstm_vae_data(df, year_loop_start, year_loop_end):
    # use only the 6 columns below and standardize them with the previous year column
    data_transformed_list = []
    numeric_columns = [
        'xGD', 'xGAD', 'xptsD', 'deepD', 'ppda_attk_diff', 'ppda_def_diff', 'ppda_coef', 'oppda_coef', 'npxGD'
    ]
    for year in range(year_loop_start, year_loop_end):
        # cur_year_data_list = []
        prev_year_col = df[df['team_season'] == year - 1]
        for _team_n_yr in df['team_year'].unique():
            if str(year) in _team_n_yr:
                cur_team_year_data_list = []
                for col in numeric_columns:
                    prev_year_mu = prev_year_col[col].mean()
                    prev_year_std = prev_year_col[col].std()
                    cur_team_year_df = df[(df['team_season'] == year) & (df['team_year'] == _team_n_yr)] \
                        .set_index(['date', 'team_year', 'team_season'])[[col]]
                    cur_team_year_df[f'normalized_{col}'] = cur_team_year_df[col].apply(
                        lambda x: (x - prev_year_mu) / prev_year_std
                    )
                    cur_team_year_data_list.append(cur_team_year_df[f'normalized_{col}'])
                cur_team_year_data_list.append(df[(df['team_season'] == year) & (df['team_year'] == _team_n_yr)]
                                               .set_index(['date', 'team_year', 'team_season'])['Home/Away'].apply(
                    lambda x: 1 if x == 'h' else 0))
                cur_team_year_data = pd.concat(cur_team_year_data_list, axis=1)
                for _seq in gen_seq(cur_team_year_data, 5, cur_team_year_data.columns):
                    data_transformed_list.append(_seq)
    return np.asarray(data_transformed_list)


def vae_train_test_val_split(transformed_df, data_dict=None):
    # split the data according to train/validation/test split using 2015-2017, 2018 & 2019 data respectively
    if data_dict is None:
        data_dict = {
            'train': {},
            'val': {},
            'test': {},
        }
    data_dict['train']['x2'] = process_lstm_vae_data(transformed_df, 2015, 2018)
    data_dict['val']['x2'] = process_lstm_vae_data(transformed_df, 2018, 2019)
    data_dict['test']['x2'] = process_lstm_vae_data(transformed_df, 2019, 2020)
    return data_dict


def generate_lstm_vae_data(data_path, data_dict=None):
    df = get_data(data_path)
    transformed_df = transform_data(df)
    transformed_df_dict = vae_train_test_val_split(transformed_df, data_dict)
    return transformed_df_dict




####
# testing_area
df = generate_lstm_vae_data("data")
# df = get_data("data")
# transformed_df = transform_data(df)
# trans_numeric_df = process_lstm_vae_numerical_data(transformed_df)


# for seq in gen_seq(df, 5, df.columns):
#     sequence_input.append(seq)
#
# sequence_input = np.asarray(sequence_input)