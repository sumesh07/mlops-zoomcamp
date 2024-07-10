#!/usr/bin/env python
# coding: utf-8

# ## Q1. Notebook
# We'll start with the same notebook we ended up with in homework 1. We cleaned it a little bit and kept only the scoring part. You can find the initial notebook here.
# 
# Run this notebook for the March 2023 data.
# 
# What's the standard deviation of the predicted duration for this dataset?
# 
# - 1.24
# - 6.24
# - 12.28
# - 18.28
# 
# Answer: 6.24

# In[1]:


# get_ipython().system('pip freeze | grep scikit-learn') # type: ignore


# In[2]:


# get_ipython().system('python -V')


# In[6]:


import os
import sys
import pickle
import pandas as pd
import sklearn

# In[19]:


year = int(sys.argv[1])
month = int(sys.argv[2])

input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
output_file = f'output/yellow_tripdata_{year:04d}-{month:02d}.parquet'

MODEL_FILE = os.getenv('MODEL_FILE', 'model.bin')

# In[9]:


with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)


# In[10]:


categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


# In[13]:


# pip install pyarrow


# In[14]:


df = read_data(input_file)


# In[15]:


dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)


# In[16]:


y_pred.std()

print('predicted mean duration:', y_pred.mean())

# ## Q2. Preparing the output
# Like in the course videos, we want to prepare the dataframe with the output.
# 
# First, let's create an artificial ride_id column:
# 
# df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
# Next, write the ride id and the predictions to a dataframe with results.
# 
# Save it as parquet:
# 
# df_result.to_parquet(
#     output_file,
#     engine='pyarrow',
#     compression=None,
#     index=False
# )
# What's the size of the output file?
# 
# - 36M
# - 46M
# - 56M
# - 66M Note: Make sure you use the snippet above for saving the file. It should contain only these two columns. For this question, don't change the dtypes of the columns and use pyarrow, not fastparquet.
# 
# Answer: 66M

# In[17]:


# get_ipython().system('mkdir output')




# In[20]:


df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')


# In[21]:


df_result = pd.DataFrame()
df_result['ride_id'] = df['ride_id']
df_result['predicted_duration'] = y_pred

os.makedirs('output', exist_ok=True)


# In[22]:


df_result.to_parquet(
    output_file,
    engine='pyarrow',
    compression=None,
    index=False
)


# In[23]:


# get_ipython().system('ls -lh output')


# ## Q3. Creating the scoring script
# Now let's turn the notebook into a script.
# 
# Which command you need to execute for that?

# 
