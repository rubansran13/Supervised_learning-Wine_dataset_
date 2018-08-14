# Predictive_analysis_of_Wine_dataset_

Wine data set(available at University of California Machine Learning Repository:https://archive.ics.uci.edu/ml/datasets/Wine+Quality) is analyzed using regression models. Dataset consists of 4898 instances and 12 variables with tareget variable of wine quality. Dataset is available for Red Wine with 6497 instances and White Wine with 4898 instances, which have been combined and color of wine is also used as a feature for estimation of wine quality.

To check presence of multicolinearity Variance Inflation Analysis is done, which shows high Variance Inflation Factor(VIF) for density that of 21 and few other variable have VIF near 10, thus requiring action and for which Principle Component Analysis(PCA) was performed.

Different Linear Methods were applied-Linear Regression,Lasso Regression,Ridge Regression and Elastic Net. Along with thse regression analysis was performed by using Random Forest Regression,Support Vector Regression and Logistic Regression.

Hyperparameter for models used are selected through cross validation and regression is performed on the data that is transformed into Principle Components.

Of all models Random Forest Regression provided the best estimates.

Refrences:
P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. 
Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.

Libreries Used: 
Scikit-learn->0.19.1
Pandas->0.23.3
Numpy->1.14.5
Statsmodels->0.9.0
Matplotlib->2.2.2
