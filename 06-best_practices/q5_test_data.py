import os
import sys
from datetime import datetime
import pandas as pd

def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)


options = {
    'client_kwargs': {
        'endpoint_url': "http://localhost:4566"
    }
}
def create_data():
    year = int(sys.argv[1])
    month = int(sys.argv[2])
    data = [
        (None, None, dt(1, 1), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2, 0), dt(1, 2, 59)),
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),      
     ]

    columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
    df_input = pd.DataFrame(data, columns=columns)

    df_input.to_parquet(
		f"s3://nyc-duration/in/{year:04d}-{month:02d}.parquet",
		engine="pyarrow",
		compression=None,
		index=False,
		storage_options=options,
	)

    

	


if __name__ == "__main__":
    create_data()