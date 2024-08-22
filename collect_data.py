import pandas as pd
import os
import fastf1
from joblib import dump


from sklearn.preprocessing import MinMaxScaler

race = 'Spanish Grand Prix'
sessions = ['FP1', 'FP2', 'FP3', 'Qualifying']
year = 2024
threshold = 1.15
filename = f'{race} {year} pre-race data.csv'
features = ['Session', 'LapTime_ms', 'LapNumber', 'Stint', 'Sector1Time_ms',
            'Sector2Time_ms', 'Sector3Time_ms', 'SpeedI1', 'SpeedI2',
            'SpeedFL', 'SpeedST', 'Compound', 'TyreLife', 'Driver',
            'Team', 'Sector1Time_DecayRate_ms', 'Sector2Time_DecayRate_ms', 'Sector3Time_DecayRate_ms']

def calculate_decay_rate(driver_group, sector_col):
    # Find the driver's personal best sector time and the corresponding lap number
    personal_best_sector_time = driver_group[sector_col].min()
    baseline_lap = driver_group['LapNumber'][driver_group[sector_col] == personal_best_sector_time].values[0]

    # Calculate the decay rate for each sector
    def decay_rate(row):
        lap_diff = row['LapNumber'] - baseline_lap
        if lap_diff != 0:
            return (row[sector_col] - personal_best_sector_time) / lap_diff
        else:
            return pd.Timedelta(0)  # Set decay rate to 0 for the benchmark lap

    driver_group[f'{sector_col}_DecayRate'] = driver_group.apply(decay_rate, axis=1)
    return driver_group

if not os.path.exists('./' + filename):
    dataframes = []
    for sess in sessions:
        session = fastf1.get_session(year, race, sess)
        session.load()
        laps = session.laps
        laps = laps[laps['PitOutTime'].isna() & laps['PitInTime'].isna()]
        laps = laps[laps['Deleted'] == False]
        laps = laps[laps['IsAccurate'] == True]

        # Filter for push laps
        driver_fastest_lap = laps.groupby('Driver')['LapTime'].min().reset_index()
        driver_fastest_lap.columns = ['Driver', 'FastestLapTime']
        laps = laps.merge(driver_fastest_lap, on='Driver')
        laps['IsPushLap'] = laps['LapTime'] <= (laps['FastestLapTime'] * threshold)
        laps = laps[laps['IsPushLap']]
        laps['Session'] = sess

        # Apply decay rate calculation for each sector
        for sector in ['Sector1Time', 'Sector2Time', 'Sector3Time']:
            laps = laps.groupby('Driver', group_keys=False).apply(calculate_decay_rate, sector)

        dataframes.append(laps)

    df = pd.concat(dataframes)

    # Convert identified time-related columns to milliseconds
    time_columns = ['LapTime', 'Sector1Time', 'Sector2Time', 'Sector3Time', 'Sector1Time_DecayRate', 'Sector2Time_DecayRate', 'Sector3Time_DecayRate']
    for col in time_columns:
        df[col] = pd.to_timedelta(df[col])
        df[col + '_ms'] = df[col].dt.total_seconds() * 1000

    # Optionally drop the original time columns if you only need the millisecond columns
    df.drop(columns=time_columns, inplace=True)

    df = df[features]
    df = df[df['SpeedI1'].notnull() & df['SpeedI2'].notnull() & df['SpeedFL'].notnull() & df['SpeedST'].notnull()]
    df.to_csv(filename, index=False)

df = pd.read_csv(filename)
print(df)

# preprocess the data:
# change the drivers, teams and session to numerical values.
compound_mapping = {
    'SOFT': 0,
    'MEDIUM': 1,
    'HARD': 2,
    'INTERMEDIATE': 3,
    'WET': 4
}
df['Team'] = pd.factorize(df['Team'])[0]
df['Driver'] = pd.factorize(df['Driver'])[0]
df['Compound'] = df['Compound'].replace(compound_mapping)
df.drop(columns='Session', inplace=True)


Y = df[['Sector1Time_DecayRate_ms', 'Sector2Time_DecayRate_ms', 'Sector3Time_DecayRate_ms']]
X = df.drop(columns=['Sector1Time_DecayRate_ms', 'Sector2Time_DecayRate_ms', 'Sector3Time_DecayRate_ms'])

XScaler = MinMaxScaler()
YScaler = MinMaxScaler()

XScaled = pd.DataFrame(XScaler.fit_transform(X), columns=X.columns)
YScaled = pd.DataFrame(YScaler.fit_transform(Y), columns=Y.columns)
XScaled.to_csv('x-preprocessed-data.csv', index=False)
YScaled.to_csv('y-preprocessed-data.csv', index=False)
dump(XScaler, 'XScaler.joblib')
dump(YScaler, 'YScaler.joblib')


