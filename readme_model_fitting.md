# Model building and tuning
> Here I will explain how to use **tuned** models to train and fit to evaluate their performance!

## Table of contents
* [Alias dictionary](#general-info)
* [Curves plotting](#baseline-model)
* [Model fitting and evaluation](#technologies)

# Baseline estimator
# Mean: 0.59, MLR: 0.13, RCV: 0.13

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


XGB_7: rmse to beat: 10.06
KNN_7: rmse = 7 in cv but 11.42 in LOOCV! NOT useful!!!

model_xgb_5_1: rmse= [cv: 0.3195 , LOOCV: 9.15]


# removing outliers
model_xgb_5_1 improves rmse from 9.2 to 8.9 after removing outliers using EllipticEnvelope(contamination=0.01). Models were tuned using GroupKFOld CV (cv=5) and were evaluated per las file. 

After tuning models with KFold CV (cv=5), model_xgb_5_1 rmse became: [8.863642255035646, 9.833146865983155], 'rmse_CV': 0.296437.

Now let's tune the models with KFold CV using contamination=0.05 for removing outliers using EllipticEnvelope, model_xgb_5_1 rmse became: 
[9.03451436065921, 10.114004938425541], 'rmse_CV': 0.27013096226469396.

Let's try contamination=0.03, 'rmse_LOOCV_mean': 8.953640123151171, 'rmse_LOOCV_corr': 9.986884226154611}, 'rmse_CV': 0.28089574167504056, 

6_2_base: rmse= [8.90, 9.83]

6_2_despike: rmse = [9.07, 10.01], it seems despike decrease performance!


with 5 clusters, 6_2_5clustered rmse=[7.02, 7.64]!!! 
rerun the 6_2_base with the same test_list as 6_2_5clustered, rmse=[7.73, 8.84].
Still 5 clusters, improved performance!

After use all scaling for all zones, the rmse for 6_2_5clustered_samescaling: [6.91, 7.53]!!! improved by [0.11, 0.11]

Now let's try 3 clusters! As expected, rmse becomes [7.89, 9.12], getting worse. 

Use 5 clusters instead!

Used two layer MLP, rmse=[6.46, 7.17]. It could be overfitting !!! So need to separate test data from training data!
But first let's do a model stack, using XGB and MLP! Stack model rmse=[6.65, 7.42]. Even though it's slightly higher than MLP, I believe it's more robust than MLP, which could have been overfitted!

Let's use stacked model, average XGB and MLP, scaling all data, then cluster the data with KMeans into 5 clusters. Build XGB and MLP models for each zone.

