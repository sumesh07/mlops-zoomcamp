import pandas as pd
from datetime import datetime
import numpy
pd.set_option('future.no_silent_downcasting', True)
 
def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)
 
columns=['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
categorical = ['PULocationID', 'DOLocationID']
 
data = [
        (None, None, dt(1, 1), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2, 0), dt(1, 2, 59)),
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),      
    ]
 
 
#   PULocationID DOLocationID tpep_pickup_datetime tpep_dropoff_datetime  duration
# 0           -1           -1  2022-01-01 01:01:00   2022-01-01 01:10:00       9.0
# 1            1            1  2022-01-01 01:02:00   2022-01-01 01:10:00       8.0
 
def prepare_data(data,columns,categorical):
    data = numpy.transpose(data)
    df1 = pd.DataFrame(data, columns)
    df = df1.T
    df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'], errors='coerce')
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'], errors='coerce')
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
 
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    df['duration']=df['duration'].astype(float)
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    print(df)
    return df
 
if __name__ == '__main__':
    columns=['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
    categorical = ['PULocationID', 'DOLocationID']
 
    data = [
        (None, None, dt(1, 1), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2, 0), dt(1, 2, 59)),
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),      
    ]
 
    input_path = f'/workspaces/mlops-zoomcamp/06-best_practices/tests/2023-01.parquet'
    df = prepare_data(data,columns,categorical)
    df.to_parquet(
        input_path,
        engine='pyarrow',
        compression=None,
        index=False
        )
 
def test():
    df_actual = prepare_data(data,columns,categorical)
    print(df_actual)
 
    data_expected = [
        ('-1', '-1', 9.0),
        ('1',  '1', 8.0),
    ]
 
    columns_test = ['PULocationID', 'DOLocationID', 'duration']
    df = numpy.transpose(data_expected)
    df1 = pd.DataFrame(df, columns_test)
    df_expected = df1.T
    df_expected['duration']=df_expected['duration'].astype(float)
    df_expected[categorical] = df_expected[categorical].fillna(-1).astype('int').astype('str')
    print("df_expected:")
    print(df_expected)
 
    print("df_actual:")
    print(df_actual)
 
    assert (df_actual['PULocationID'] == df_expected['PULocationID']).all()
    assert (df_actual['DOLocationID'] == df_expected['DOLocationID']).all()
    assert (df_actual['duration'] - df_expected['duration']).abs().sum() < 0.0000001
 
    print("df_expected:")
    print(df_expected)