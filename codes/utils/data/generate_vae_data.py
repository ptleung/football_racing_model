import pandas as pd
from codes.utils.data.get_data import get_data


def process_vae_numerical_data(df):
    # use only the 6 columns below and standardize them with the previous year column
    data_transformed_list = []
    for year in range(2015, 2020):
        prev_year_col = df[df['team_season'] == year - 1]
        cur_year_data_list = []
        for col in ['xG', 'xGA', 'xpts', 'npxG', 'npxGA', 'npxGD']:
            prev_year_mu = prev_year_col[col].mean()
            prev_year_std = prev_year_col[col].std()
            # GW>5 as the lstm model requires past 5 games as input
            cur_year_df = df[(df['team_season'] == year) & (df['GW'] > 5)].set_index(
                ['date', 'team_year', 'team_season'])[[col]]
            cur_year_df[f'normalized_{col}'] = cur_year_df[col].apply(
                lambda x: (x-prev_year_mu)/prev_year_std
            )
            cur_year_data_list.append(cur_year_df[f'normalized_{col}'])
        data_transformed_list.append(pd.concat(cur_year_data_list, axis=1))
    data_transformed = pd.concat(data_transformed_list)
    return data_transformed[[_col for _col in data_transformed.columns if "normalized_" in _col]]


def process_vae_categorical_data(df):
    # use only the `Home/Away` as categorical column
    df = df[(df['team_season'] > 2014) & (df['GW'] > 5)]  # GW>5 as the lstm model requires past 5 games as input
    df['home_game'] = df['Home/Away'].apply(lambda x: 1 if x == 'h' else 0)
    df = df.set_index(['date', 'team_year', 'team_season'])
    return df['home_game']


def vae_train_test_val_split(transformed_df, data_dict=None):
    # split the data according to train/validation/test split using 2015-2017, 2018 & 2019 data respectively
    if data_dict is None:
        data_dict = {
            'train': {},
            'val': {},
            'test': {},
        }
    data_dict['train']['x1'] = transformed_df[transformed_df.index.get_level_values(2) < 2018].values
    data_dict['val']['x1'] = transformed_df[transformed_df.index.get_level_values(2) == 2018].values
    data_dict['test']['x1'] = transformed_df[transformed_df.index.get_level_values(2) == 2019].values
    return data_dict


def generate_vae_data(data_path, data_dict=None):
    df = get_data(data_path)
    numeric_df = process_vae_numerical_data(df)
    categorical_df = process_vae_categorical_data(df)
    transformed_df = pd.concat([numeric_df, categorical_df], axis=1)
    transformed_df_dict = vae_train_test_val_split(transformed_df, data_dict)
    return transformed_df_dict
