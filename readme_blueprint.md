
# SPE-GCS Machine Learning Challenge 2021

## How to use this repository:

a) clone this repository

b) create a 'data/las/' folder and store all .las files in the folder. The las files are too large to upload to github.

c) run the load.py to create cleaned data files store in 'data/' folder.

d) run main.py to train and predict. Default to select and train on 7 features.

You can import functions from plot.py and util.py for your own use. Install required library accordingly.

## Project blue-print and tasks

a) Explore data to answer below questions

    1) Which logs to use as predictors?  Which logs are available in test las? (porosity, gamma ray, resisitivity, compressional slowness) 
        Answer from committee: not sure. Possible input: [DTC], [DTC, RHOB], [DTC, RHOB, RESISTIVITY/GAMMA RAY/POROSITY],
    
    2) Which are duplciate logs with different mnemonics but for the same log? (e.g. resistivity logs) What to do with them?
        Solution: harmonize the data, create dictionary to achieve consistent mnemonics and units.

    Outputs: 
    1) Dataset(s) in pandas.DataFrame with organized predictors (DTCO etc.) and response (DTSM). (we won't split to train/test as we will do cross validation for model evaluation)

    2) A function to process any las file to output desired predictors

b) Test regression algorithms to answer below questions

    0) Baseline model: DTS/DTC ratio, two empirical models (Castagna et al. 1985, Han et al. 1986)

    1) Which scikit-learn works best based on rmse? Focus on SVR 
        linear regression, 
        ridge regression, lass regression, 
        support vector regression, 
        gradient boosting, XGB, 
        random forest etc.

    2) Which neural network algorithm works best based on rmse? 
        Try both c-RNN, ANN first as recommended in the paper

    3) Would model stacking improve the model beyond simple average of predictions?

    Outputs:
    1) Model/algorithm 
    2) Predictions vs. actual data in Excel

c) Potential data problems
    
    1) Various log resolution (sync resolution)
    2) GR not helpful? How about resistivity? 
    3) Different depth/formations have different model? 
    4) Misssing values? 
    5) Anomaly values?
    6) Correct units

d) Coding format

    1) Modularize your code by writing a function if possible
    2) Follow sci-kit learn style if possible

## Project timeline

    Committee schedule
    • 15th January 11:30 AM – 1:00 PM Kick off Session
    • 22nd January 12:00 Noon – 1:00 PM Data Engineering (Babak Akbari)
            `Team goal: complete data exploration`
    • 29th January 12:00 Noon – 1:00 PM Data Modeling (Sunil Garg/Anisha Kaul)
    • ### 29th January: 10 well logs data will be released for testing purposes and leaderboard
            `Team goal: complete initial algorithm testing and submit predictions in Excel`
    • 1st February: First Leaderboard
    • 5th February 12:00 Noon – 1:00 PM ML Strategy & Parameter Tuning (Shivam Agarwal)
    • 7th February: Second Leaderboard
    • ### 12th February: Complete testing dataset (20 well logs data) will be released
            `Team goal: complete initial algorithm tuning and submit predictions in Excel`
    • ### 14th February: Final Leaderboard
            `Team goal: complete final algorithm tuning and submit github and predictions in Excel`
    • 15th February 5:00 PM Final Code & Presentation DUE
            'Team goal: submit video presentation'            
    • 22nd February winners will be announced



# .py manual

main.py - build the model then TRAIN the model and predict
main_evaluate.py - load the already trained model to predict
mian_test.py - load the already trained model to predict on test data (test data for leaderboard 1, 2 etc...)
plot.py - plot 