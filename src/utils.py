
import pandas as pd

#f1_data_prep için
def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe #  #f1 data prep çağrıyor

team_name_map = {
    'toro_rosso': 'rb',
    'alphatauri': 'rb',
    'force_india': 'aston_martin',
    'racing_point': 'aston_martin',
    'renault': 'alpine',
    'lotus_f1': 'alpine',
    'alfa': 'sauber',
    'audi': 'sauber',
    'lotus_racing': 'caterham',
    'virgin': 'manor',
    'marussia': 'manor',
}



# 4.1_feature_engineering.py'daki yedi özelliğin fonksiyon hali. Yeni bir
# yarışın satırları geçmiş veriye eklenip buradan geçirilmeli; rolling ve
# expanding hesaplar ancak böyle gerçek geçmişe bakabilir.
def add_features(dataframe):
    df = dataframe.sort_values(['driver_id', 'season', 'round']).reset_index(drop=True)

    df['team_unified'] = df['constructor_id'].map(team_name_map).fillna(df['constructor_id'])

    df['rolling_form'] = df.groupby('driver_id')['position'].transform(
        lambda s: s.shift(1).rolling(3).mean())

    df['season_points'] = (
        df.groupby(['driver_id', 'season'])['points']
        .transform(lambda s: s.shift(1).cumsum())
        .fillna(0)
    )

    team = (df.groupby(['team_unified', 'season', 'round'], as_index=False)['position'].mean()
               .sort_values(['team_unified', 'season', 'round']))
    team['team_form'] = team.groupby('team_unified')['position'].transform(
        lambda s: s.shift(1).rolling(3).mean())
    df = df.merge(team[['team_unified', 'season', 'round', 'team_form']],
                  on=['team_unified', 'season', 'round'], how='left')
    df = df.sort_values(['driver_id', 'season', 'round']).reset_index(drop=True)

    df['circuit_history'] = df.groupby(['driver_id', 'circuit_id'])['position'].transform(
        lambda s: s.shift(1).expanding().mean())

    car_count = df.groupby(['team_unified', 'season', 'round'])['position'].transform('size')
    team_avg = df.groupby(['team_unified', 'season', 'round'])['position'].transform('mean')
    diff = (2 * team_avg - df['position']) - df['position']
    df['_diff'] = diff.where(car_count == 2)
    df['teammate_delta'] = df.groupby('driver_id')['_diff'].transform(
        lambda s: s.shift(1).expanding().mean())

    finished = (df['status'].eq('Finished')
                | df['status'].str.startswith('+', na=False)
                | df['status'].eq('Lapped'))
    df['_dnf'] = (~finished).astype(int)
    df['dnf_rate'] = df.groupby('driver_id')['_dnf'].transform(
        lambda s: s.shift(1).expanding().mean())

    df['_gain'] = df['grid'] - df['position']
    df['grid_gain'] = df.groupby('driver_id')['_gain'].transform(
        lambda s: s.shift(1).expanding().mean())

    df['career_races'] = df.groupby('driver_id').cumcount()

    derived_cols = ['rolling_form', 'season_points', 'team_form', 'circuit_history',
                     'teammate_delta', 'dnf_rate', 'grid_gain', 'career_races']
    for col in derived_cols:
        df[col] = df[col].fillna(df[col].median())

    return df.drop(columns=[c for c in df.columns if c.startswith('_')])

#-------------------------------------

# Modele girecek X ile hedef y'yi ayırır: sızıntı ve kimlik sütunlarını
# atar, team_unified'ı one-hot'a çevirir. Girdi add_features'tan geçmiş
# bir tablo olmalı.
def f1_data_prep(dataframe):
    leak_cols = ["position", "position_text", "points", "laps", "status"]
    id_cols = ["driver_id", "constructor_id", "circuit_id", "race_name", "date"]
    drop_cols = leak_cols + id_cols + ["season", "is_podium"]

    y = dataframe["is_podium"]
    X = dataframe.drop(columns=drop_cols)
    X = one_hot_encoder(X, ["team_unified"], drop_first=True)

    return X, y
