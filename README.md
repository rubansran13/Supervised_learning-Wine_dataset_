# Predictive_analysis_of_Wine_dataset_

Wine data set(available at University of California Machine Learning Repository:https://archive.ics.uci.edu/ml/datasets/Wine+Quality) is analyzed using regression and classification models. Dataset has 6497 instances, of which Red Wine dataset has   instances and White Wine with 4898 instances, which have been combined and color of wine is also used as a feature for estimation of wine quality.

The data is split into train set and test set in 3:1 ratio. To avoid any snooping into test set, it is not used for any analysis. To check presence of multicolinearity, first a correlation heatmap is drawn on the train set.

![Alt Text](Wine_Characteristics_Correlation_Heatmap(Train_set).png?raw=true "Correlation Heatmap")

Several features shows higher degree of correlaiton-free sulfur dixide eith total sulfur dioxide; total sulfur dioxide, free sulfur dixide and density with residual sugar. Thus Variance Inflation Analysis is done to know the degree of multicollinearity. THis analysis shows high Variance Inflation Factor(VIF) for density that of 21 and few other variable have VIF near 10, thus requiring action for dealing with multicollinearity and thus, Principle Component Analysis(PCA) was performed.

![Alt Text](Variance_Inflation_Factor.png?raw=true "Checking Multicolinearity")


![Alt Text](PCA_Variance_Explained.png?raw=true "Variance explained by Principle Components")
The first 10 principle components explains ~99 percent of the variance. Thus the standardized training set and test set are transformed using thses 10 components

Different Linear Methods were applied-Linear Regression,Lasso Regression,Ridge Regression and Elastic Net. Along with thse regression analysis was performed by using Random Forest Regression,Support Vector Regression and Logistic Regression. Accuracy Scores are also included in regression model performance along with mean squared error and r2 score. Accuracy scores were calculated using rounding function where (X.5,Y.5] was converted to Y, where X and Y are integers and Y=X+1. Using rounding, calculating accuracy score was possible as it reduced the regression predicted values to integer values which could be treated as classes of classification.

For classification Gaussian Naive Bayes, Decision Tree, K Nearest Neighbours and Random Forest Classifier are used.

Hyperparameter selection for models is done through cross validation and analysis is performed on the data that is transformed into Principle Components.

Through regression and classification analysis, Random Forest give the best results with accuracy of 0.68. In classification analysis K Nearest Neighbours performs on the on par with Random Forest with accuracy of 0.68. Other model fall short of 0.60 accuracy mark.

Refrences:
P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. 
Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.

Libreries Used: 
Scikit-learn->0.19.1
Pandas->0.23.3
Numpy->1.14.5
Statsmodels->0.9.0
Matplotlib->2.2.2
Seaborn->0.7.1
