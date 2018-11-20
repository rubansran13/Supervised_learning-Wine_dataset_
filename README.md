# Predictive_analysis_of_Wine_dataset_

Wine data set(available at University of California Machine Learning Repository:https://archive.ics.uci.edu/ml/datasets/Wine+Quality) is analyzed using regression and classification models. Dataset has 6497 instances, of which Red Wine dataset has  1599 instances and White Wine with 4898 instances, which have been combined and color of wine is also used as a feature for estimation of wine quality.


![Alt Text](Wine_Quality_Histogram.png?raw=true "Histogram Wine Quality")

Wine quality target presents the problem of imbalanced data where wine quality mapped to 3 and 9 are minority classes. To deal with this, the problem is reduced to three class classification problem. All the targets having a score less than 6 are mapped to 0, equal to 6 are mapped to 1 and greater than 6 are mapped to 2.

The data is split into train set and test set in 3:1 ratio. To avoid any snooping into test set, it is not used for any analysis. To check presence of multicolinearity, first a correlation heatmap is drawn on the train set.

![Alt Text](Wine_Characteristic_Correlation Heatmap(Train set).png?raw=true "Correlation Heatmap")


Several features shows higher degree of correlaiton-free sulfur dixide eith total sulfur dioxide; total sulfur dioxide, free sulfur dixide and density with residual sugar. Thus Variance Inflation Analysis is done to know the degree of multicollinearity. THis analysis shows high Variance Inflation Factor(VIF) for density that of 21 and few other variable have VIF near 10, thus requiring action for dealing with multicollinearity and thus, Principle Component Analysis(PCA) was performed.


![Alt Text](Variance_Inflation_Factor.png?raw=true "Checking Multicolinearity")


![Alt Text](PCA_Variance_Explained.png?raw=true "Variance explained by Principle Components")


The first 10 principle components explains ~99 percent of the variance. Thus the standardized training set and test set are transformed using thses 10 components


For classification Gaussian Naive Bayes,Logistic Regression, Decision Tree, K Nearest Neighbours, Random Forest Classifier and Support Vector Classification are used. Hyperparameter selection for models is done through cross validation and analysis is performed on the data that is transformed into Principle Components.

Classification analysis, Random Forest and K Nearest Neighbours give the best results with accuracy of 0.71 on three class classification problem. 

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
