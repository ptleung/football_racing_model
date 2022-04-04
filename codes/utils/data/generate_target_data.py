from codes.utils.data.get_data import get_data


def process_target_data(df):
    # Use convert the W/D/L data into a 3 class target output df
    # GW>5 as the lstm model requires past 5 games as input
    df = df[(df['team_season'] > 2014) & (df['GW'] > 5)]
    for _result in ['w', 'd', "l"]:
        df[f'result_{_result}'] = df['result'].apply(lambda x: 1 if x == _result else 0)
    df = df.set_index(['date', 'team_year', 'team_season'])
    return df[[col for col in df.columns if "result_" in col]]


def target_train_test_val_split(transformed_df, data_dict=None):
    # split the data according to train/validation/test split using 2015-2017, 2018 & 2019 data respectively
    if data_dict is None:
        data_dict = {
            'train': {},
            'val': {},
            'test': {},
        }
    data_dict['train']['y'] = transformed_df[transformed_df.index.get_level_values(2) < 2018].values
    data_dict['val']['y'] = transformed_df[transformed_df.index.get_level_values(2) == 2018].values
    data_dict['test']['y'] = transformed_df[transformed_df.index.get_level_values(2) == 2019].values
    return data_dict


def generate_target_data(data_path, data_dict):
    df = get_data(data_path)
    target_df = process_target_data(df)
    target_df_dict = target_train_test_val_split(target_df, data_dict)
    return target_df_dict
