# -*- coding: utf-8 -*-
"""
Created on Mon May 16 16:11:19 2022

@author: a454g185
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import missingno as mno

import warnings
warnings.filterwarnings("ignore")
import logging
logging.disable(logging.CRITICAL)


from darts import TimeSeries, concatenate
from darts.dataprocessing.transformers import Scaler
from darts.models import TransformerModel
from darts.metrics import mape, rmse
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.utils.likelihood_models import QuantileRegression

data  = pd.read_csv(r'https://raw.githubusercontent.com/armangh67/LSTM/main/data/SolarData.csv')
data['Date'] = pd.to_datetime(data['Date'], utc = True)

df = data[['Date','Power', 'Temp', 'radiation_direct','radiation_h']]

############# Filling null values ###############
dataset = df.set_index('Date')
dataset = dataset.fillna(0)


######## Plot couple of Days ##########################
plot = dataset.head(288)
plt.figure(figsize=(25,6))
plt.rcParams.update({"font.size" : 18})
plt.figure(figsize= (20,8))
plt.plot(plot['Power'],  label = 'PV Generation',linewidth=2)
plt.xlabel('Date')
plt.ylabel('Solar Power (MW)')
plt.title("Actual solar generation in  Denmark(TransnetBW) in MW")
plt.grid(True)
plt.legend()
plt.show()



LOAD = False         # True = load previously saved model from disk?  False = (re)train the model
SAVE = "\_TForm_model10e.pth.tar"   # file name to save the model under

EPOCHS = 800
INLEN = 4         # input size
FEAT = 8           # d_model = number of expected features in the inputs, up to 512    
HEADS = 4           # default 8
ENCODE = 4          # encoder layers
DECODE = 4          # decoder layers
DIM_FF = 64       # dimensions of the feedforward network, default 2048
BATCH = 8          # batch size
ACTF = "relu"       # activation function, relu (default) or gelu
SCHLEARN = None     # a PyTorch learning rate scheduler; None = constant rate
LEARN = 1e-3        # learning rate
VALWAIT = 1         # epochs to wait before evaluating the loss on the test/validation set
DROPOUT = 0.1       # dropout rate
N_FC = 1            # output size

RAND = 42           # random seed
N_SAMPLES = 3     # number of times a prediction is sampled from a probabilistic model
N_JOBS = 3          # parallel processors to use;  -1 = all processors

# default quantiles for QuantileRegression
QUANTILES = [0.01, 0.1, 0.2, 0.5, 0.8, 0.9, 0.99]

SPLIT = 0.7         # train/test %

FIGSIZE = (9, 6)


qL1, qL2 = 0.01, 0.10        # percentiles of predictions: lower bounds
qU1, qU2 = 1-qL1, 1-qL2,     # upper bounds derived from lower bounds
label_q1 = f'{int(qU1 * 100)} / {int(qL1 * 100)} percentile band'
label_q2 = f'{int(qU2 * 100)} / {int(qL2 * 100)} percentile band'

mpath = os.path.abspath(os.getcwd()) + SAVE     # path and file name to save the model

df2 = dataset.head(20000)
df2.head(20)


# check correlations of features with price
df_corr = df2.corr(method="pearson")
print(df_corr.shape)
print("correlation with Wind Power:")
df_corrP = pd.DataFrame(df_corr["Power"].sort_values(ascending=False))
df_corrP

# highest absolute correlations with price
pd.options.display.float_format = '{:,.2f}'.format
df_corrH = df_corrP[np.abs(df_corrP["Power"]) > 0.25]
df_corrH


df3 = df2[df_corrH.index]
df3.head()


# additional datetime columns: feature engineering
df3["month"] = df3.index.month

df3["wday"] = df3.index.dayofweek
dict_days = {0:"1_Mon", 1:"2_Tue", 2:"3_Wed", 3:"4_Thu", 4:"5_Fri", 5:"6_Sat", 6:"7_Sun"}
df3["weekday"] = df3["wday"].apply(lambda x: dict_days[x])

df3["hour"] = df3.index.hour

df3 = df3.astype({"hour":float, "wday":float, "month": float})

df3.iloc[[0, -1]]


# pivot table: weekdays in months
piv = pd.pivot_table(   df3, 
                        values="Power", 
                        index="month", 
                        columns="weekday", 
                        aggfunc="mean", 
                        margins=True, margins_name="Avg", 
                        fill_value=0)
pd.options.display.float_format = '{:,.0f}'.format

# dataframe with price and features only
df4 = df3.copy()
df4.drop(["weekday", "month", "wday", "hour"], inplace=True, axis=1)

# create time series object for target variable
ts_P = TimeSeries.from_series(df4["Power"]) 

# check attributes of the time series
print("components:", ts_P.components)
print("duration:",ts_P.duration)
print("frequency:",ts_P.freq)
print("frequency:",ts_P.freq_str)
print("has date time index? (or else, it must have an integer index):",ts_P.has_datetime_index)
print("deterministic:",ts_P.is_deterministic)
print("univariate:",ts_P.is_univariate)

# create time series object for the feature columns
df_covF = df4.loc[:, df4.columns != "Power"]
ts_covF = TimeSeries.from_dataframe(df_covF)

# check attributes of the time series
print("components (columns) of feature time series:", ts_covF.components)
print("duration:",ts_covF.duration)
print("frequency:",ts_covF.freq)
print("frequency:",ts_covF.freq_str)
print("has date time index? (or else, it must have an integer index):",ts_covF.has_datetime_index)
print("deterministic:",ts_covF.is_deterministic)
print("univariate:",ts_covF.is_univariate)

# example: operating with time series objects:
# we can also create a 3-dimensional numpy array from a time series object
# 3 dimensions: time (rows) / components (columns) / samples
ar_covF = ts_covF.all_values()
print(type(ar_covF))
ar_covF.shape

# example: operating with time series objects:
# we can also create a pandas series or dataframe from a time series object
df_covF = ts_covF.pd_dataframe()
type(df_covF)

# train/test split and scaling of target variable
ts_train, ts_test = ts_P.split_after(SPLIT)
print("training start:", ts_train.start_time())
print("training end:", ts_train.end_time())
print("training duration:",ts_train.duration)
print("test start:", ts_test.start_time())
print("test end:", ts_test.end_time())
print("test duration:", ts_test.duration)

scalerP = Scaler()
scalerP.fit_transform(ts_train)
ts_ttrain = scalerP.transform(ts_train)
ts_ttest = scalerP.transform(ts_test)    
ts_t = scalerP.transform(ts_P)

# make sure data are of type float
ts_t = ts_t.astype(np.float32)
ts_ttrain = ts_ttrain.astype(np.float32)
ts_ttest = ts_ttest.astype(np.float32)

print("first and last row of scaled price time series:")
pd.options.display.float_format = '{:,.2f}'.format
ts_t.pd_dataframe().iloc[[0,-1]]

# train/test split and scaling of feature covariates
covF_train, covF_test = ts_covF.split_after(SPLIT)

scalerF = Scaler()
scalerF.fit_transform(covF_train)
covF_ttrain = scalerF.transform(covF_train) 
covF_ttest = scalerF.transform(covF_test)   
covF_t = scalerF.transform(ts_covF)  
covF_t = covF_t.astype(np.float32)

# make sure data are of type float
covF_ttrain = ts_ttrain.astype(np.float32)
covF_ttest = ts_ttest.astype(np.float32)

pd.options.display.float_format = '{:.2f}'.format
print("first and last row of scaled feature covariates:")
covF_t.pd_dataframe().iloc[[0,-1]]


# feature engineering - create time covariates: hour, weekday, month, year, country-specific holidays
covT = datetime_attribute_timeseries(ts_P.time_index, attribute="hour", one_hot=False)
covT = covT.stack(datetime_attribute_timeseries(ts_P.time_index, attribute="day_of_week", one_hot=False))
covT = covT.stack(datetime_attribute_timeseries(ts_P.time_index, attribute="month", one_hot=False))
covT = covT.stack(datetime_attribute_timeseries(ts_P.time_index, attribute="year", one_hot=False))

covT = covT.add_holidays(country_code="ES")
covT = covT.astype(np.float32)


# train/test split
covT_train, covT_test = covT.split_after(SPLIT)


# rescale the covariates: fitting on the training set
scalerT = Scaler()
scalerT.fit(covT_train)
covT_ttrain = scalerT.transform(covT_train)
covT_ttest = scalerT.transform(covT_test)
covT_t = scalerT.transform(covT)
covT_t = covT_t.astype(np.float32)


pd.options.display.float_format = '{:.0f}'.format
print("first and last row of unscaled time covariates:")
covT.pd_dataframe().iloc[[0,-1]]


# combine feature covariates and time covariates in a single time series object
ts_cov = ts_covF.concatenate(covT, axis=1)                      # unscaled F+T
cov_t = covF_t.concatenate(covT_t, axis=1)                      # scaled F+T
cov_ttrain = covF_ttrain.concatenate(covT_ttrain, axis=1)       # scaled F+T training

print("first and last row of unscaled covariates:")
ts_cov.pd_dataframe().iloc[[0,-1]]



model = TransformerModel(
                    input_chunk_length = INLEN,
                    output_chunk_length = N_FC,
                    batch_size = BATCH,
                    n_epochs = EPOCHS,
                    model_name = "Transformer_wind",
                    nr_epochs_val_period = VALWAIT,
                    d_model = FEAT,
                    nhead = HEADS,
                    num_encoder_layers = ENCODE,
                    num_decoder_layers = DECODE,
                    dim_feedforward = DIM_FF,
                    dropout = DROPOUT,
                    activation = ACTF,
                    random_state=RAND,
                    #likelihood=QuantileRegression(quantiles=QUANTILES), 
                    optimizer_kwargs={'lr': LEARN},
                    add_encoders={"cyclic": {"future": ["hour", "dayofweek", "month"]}},
                    save_checkpoints=True,
                    force_reset=True
                    )


# training: load a saved model or (re)train
if LOAD:
    print("have loaded a previously saved model from disk:" + mpath)
    model = TransformerModel.load_model(mpath)                            # load previously model from disk 
else:
    model.fit(  ts_ttrain, 
                past_covariates=cov_t, 
                verbose=True)
    print("have saved the model after training:", mpath)
    model.save_model(mpath)
    
    
# testing: generate predictions
ts_tpred = model.predict(   n=len(ts_ttest), 
                            num_samples=N_SAMPLES,   
                            n_jobs=N_JOBS, 
                            verbose=True)



dfY = pd.DataFrame()
dfY["Actual"] = TimeSeries.pd_series(ts_test)
ts_tq = ts_tpred.quantile_timeseries(1)
ts_q = scalerP.inverse_transform(ts_tq)
dfY["Preds"] = TimeSeries.pd_series(ts_q)
dfY.head()


from sklearn.metrics import mean_squared_error
RMSE1 = format(np.sqrt(mean_squared_error(dfY['Actual'],dfY['Preds'] )),".3f")
print("The RMSE valu is:",RMSE1)


import matplotlib.pyplot as plt
dfY1 = dfY.head(288)
# plot the forecast
plt.figure(100, figsize=(25, 7))
sns.set(font_scale=1.3)
p = sns.lineplot(x='Date', y = "Preds", data = dfY1)
sns.lineplot( x='Date', y = "Actual", data = dfY1)
plt.legend(labels=["Preds", "Actual Power"])
p.set_ylabel("PV Power")
p.set_xlabel("Time")
plt.show()




path=r'C:\Users\a454g185\Desktop\LSTM\Transformer\Code'

dfY.to_csv(path + 'df_'+str(EPOCHS)+'epochs'+str(BATCH)+'_batch.csv')



















