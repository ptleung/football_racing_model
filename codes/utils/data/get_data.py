import pandas as pd
import os
import ast


def get_data(path):
    data_list = []
    for subdir, dirs, files in os.walk(path):
        for file in files:
            file_path = subdir + os.sep + file
            team_year = "_".join(file.split("_")[:2])
            team_season = int(file.split("_")[1])
            if file_path.endswith(".csv"):
                _df = pd.read_csv(file_path)
                _df['team_year'] = team_year
                _df['team_season'] = team_season
                data_list.append(_df)
    df = pd.concat(data_list)
    df['ppda'] = df['ppda'].apply(ast.literal_eval)
    df['ppda_allowed'] = df['ppda_allowed'].apply(ast.literal_eval)
    return df
