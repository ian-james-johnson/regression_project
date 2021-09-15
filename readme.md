This repository holds files for my Codeup regression classification project.
The work is done in the regression_project notebook.
The accessory files needed to run the notebook are:
acquire.py, propare.py, and model.py
In addition, a env.py or zillow.csv file is needed to get the data.
To recreate the project, copy the repository to your drive and then run regression_project_v3.jpynb

Data dictionary:
bedroomcnt                        number of bedrooms
bathroomcnt                       number of bathrooms
calculatedfinishedsquarefeet      area of property (sq ft)
taxvaluedollarcnt                 tax value of property
yearbuilt                         year the property was built
taxamount                         amount of taxes for the property
fips                              geographical region of the property

The purpose of the project is to determine which features can be used to predict property values.

My initail hypothesis is that the best features are calculatedfinishedsquarefeet, bedroomcnt, bathroomcnt.

The distribution of taxes by region were explored.
Pair plots were used to explore the features.
Correlation of the features was determined.

The most predictive features were determined by KBest and RFE.
The best features are calculatedfinishedsquarefeet, bedroomcnt, bathroomcnt.

The models used were linear model, generalized linear model, and lassolars.
The models were compared to a median baseline.
Lassolars was the best performing model, as determined by RMSE and R^2.