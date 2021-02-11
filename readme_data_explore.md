# Model building and tuning
> Here I will explain how models are built and tuned for best performance!

## Table of contents
* [General info](#general-info)
* [Crossplot DTCO-DTSM](#dtco-dtsm)

## Crossplot DTCO-DTSM
![Crossplot](./readme_resources/Crossplot-DTCO-DTSM.png)


Ideas to improve modeling?
Nullify the Extreme Values or Outliers
# Limit the range of data (https://github.com/pddasig/Machine-Learning-Competition-2020/blob/master/Solutions/3_RockAbusers%20Solution%20Submission.ipynb)
df = df[(df[CNS]<1)&(df[CNS]>-0.2)&(df[RHOB]<3)&(df[RHOB]>1.75)&(df[DTC]<160)&(df[DTC]>40)]

# GR
df1['GR'][(df1['GR']>250)] = np.nan
# CNC
df1['CNC'][df1['CNC']>0.7] = np.nan
# HRM & HRD
df1['HRD'][df1['HRD']>200] = np.nan
df1['HRM'][df1['HRM']>200] = np.nan