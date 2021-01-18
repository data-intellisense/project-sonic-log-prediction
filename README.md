
# Machine Learning Challenge organized by SPE-GCS in 2021

## How to use this repository:
a) clone this repository
b) create a 'data/las/' folder and store all .las files in the folder
c) run the load.py to create a .h5 file which stores the curves and data info
d) run data_explore.py to explore data and plot logs

You can import plot_logs function from plot.py to plot the logs in your code! Plotly package needs to be installed to use the plot_logs function.

##Project blue-print and tasks:

a) Explore data to answer below questions:
    1) Which logs to use as predictors? (porosity, gamma ray, resisitivity, compressional slowness)
    2) Which logs are available in test las? (to ask/confirm competition committee)
    3) Which are duplciate logs with different mnemonics but for the same log? (e.g. resistivity logs) What to do with them?

    Output: 
    1) Dataset(s) in pandas.DataFrame with organized predictors (DTCO etc.) and response (DTSM). (we won't split to train/test as we will do cross validation for model evaluation)
    2) A function to process any las file to output desired predictors

b) Test regression algorithms to answer below questions:
    1) Which scikit-learn works best based on rmse? (linear regression, ridge regression, lass regression, support vector regression, gradient boosting, XGB, random forest etc.)

    2) Which neural network algorithm works best based on rmse? Try both c-RNN, ANN first as recommended in the paper

    3) Would model stacking improve the model beyond simple average of predictions?

    Output:
    1) Final model/algorithm 
    2) Predictions vs. actual data in Excel

##Project timeline:

    Committee schedule:
    • 15th January 11:30 AM – 1:00 PM Kick off Session
    • 22nd January 12:00 Noon – 1:00 PM Data Engineering (Babak Akbari)
            `Team goal: complete data exploration`
    • 29th January 12:00 Noon – 1:00 PM Data Modeling (Sunil Garg/Anisha Kaul)
    • ###29th January: 10 well logs data will be released for testing purposes and leaderboard
            `Team goal: complete initial algorithm testing and submit predictions in Excel`
    • 1st February: First Leaderboard
    • 5th February 12:00 Noon – 1:00 PM ML Strategy & Parameter Tuning (Shivam Agarwal)
    • 7th February: Second Leaderboard
    • ###12th February: Complete testing dataset (20 well logs data) will be released
            `Team goal: complete initial algorithm tuning and submit predictions in Excel`
    • ### 14th February: Final Leaderboard
            `Team goal: complete final algorithm tuning and submit github and predictions in Excel`
    • 15th February 5:00 PM Final Code & Presentation DUE
            'Team goal: submit video presentation'            
    • 22nd February winners will be announced




