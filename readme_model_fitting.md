# Model building and tuning
> Here I will explain how to use **tuned** models to train and fit to evaluate their performance!

## Table of contents
* [Alias dictionary](#general-info)
* [Curves plotting](#baseline-model)
* [Model fitting and evaluation](#technologies)

## General info

XGB_7_1: rmse=10.282 (108 files) with sample_weight=None, run in 850 seconds

XGB_7_1: rmse=10.184 (108 files) with sample_weight=sample_weight2, run 920 seconds

XGB_7:   rmse= 9.856 (108 files) with sample_weight=sample_weight2, run 1055 seconds

So sample_weight2 DOES improve model performance by 0.1!!! This is with 'DEPTH' added as a feature.

XGB_7 (best):   rmse= 9.021 (108 files) with sample_weight=None and RobustScaler!

XGB_7:   rmse= 9.874 (108 files) with sample_weight=sample_weight2. Depth already include so sample weight reduced accuracy? 

XGB_7:   rmse= 9.75 (108 files) with sample_weight=None and NO scaling!

XGB_7: rmse = 9.87 (108 files) with log_DTSM, sample_weight=None, scaling

XGB_7: rmse = 9.80 (108 files) with log_mnemonics = ['RT', 'NPHI'], scaling

 Bad!! Using difference to do time series prediction. 20 neighboring wells were used to predict the seleted well. rmse was way to high.

After more comprehensive RandomizedSearchCV:

model_xgb_7:   9.42 (107 wells)
model_xgb_7_1: 9.37 (107 wells)
model_xgb_7_2: 9.34 (107 wells)
model_xgb_6_1: 9.89 (107 wells)
model_xgb_6_2: 8.81 (107 wells) !!!!!
model_xgb_3_1: 10.8 (107 wells)
model_xgb_3_2: 10.6 (107 wells)



# XGB model tuning

## target_mnemonics =["DTCO", "RHOB", "NPHI", "GR", "RT", "CALI", "PEFZ"]
## new_mnomonics = ['DEPTH'] 
add gradient of ['NPHI', 'GR', 'RHOB'] is not useful

Mean-Model rmse:	0.61
MLR rmse:	0.13
RCV rmse:	0.13


XGB_7: rmse to beat: 10.06


KNN_7: rmse = 7 in tuning but 11.42 in testing!


Should I fix DTCO flat data? Yes! Look at Well 62!