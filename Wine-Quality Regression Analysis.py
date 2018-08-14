import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split as tts
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor 
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import linear_model
from matplotlib import pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor as SMVIF

#reading data from the file path
white_Wine=pd.read_csv("winequality-white.csv",sep=";")
red_Wine=pd.read_csv("winequality-red.csv",sep=";")


#combining the red and white data into one with one variable corresponding to the color of wine
white_Wine["color"]="W"
red_Wine["color"]="R"
wine=pd.concat([red_Wine,white_Wine], ignore_index=True)

#redefining columns names
wine.rename(columns={'fixed acidity': 'fixed_acidity','citric acid':'citric_acid','volatile acidity':'volatile_acidity','residual sugar':'residual_sugar','free sulfur dioxide':'free_sulfur_dioxide','total sulfur dioxide':'total_sulfur_dioxide'}, inplace=True)

#adding dummy variable for color
wine=pd.get_dummies(wine,columns=['color'])

wine.rename(columns={'fixed acidity': 'fixed_acidity','citric acid':'citric_acid','volatile acidity':'volatile_acidity','residual sugar':'residual_sugar','free sulfur dioxide':'free_sulfur_dioxide','total sulfur dioxide':'total_sulfur_dioxide','color_R':'color'}, inplace=True)

#independent variables that will be used to predict quality of wine
features=['fixed_acidity','volatile_acidity','citric_acid','residual_sugar','chlorides','free_sulfur_dioxide','total_sulfur_dioxide','density','pH','sulphates','alcohol','color']

#separating independent and dependent variable
wine_features=wine[features]
wine_target=wine['quality']

#train data and test data split  (75% train data-25% test data)
X_train, X_test, Y_train,Y_test=tts(wine_features,wine_target,test_size=0.25,random_state=4,stratify =wine_target)

#removing mean and scaling to unit variance 
processing=preprocessing.StandardScaler()
X_train_std= processing.fit_transform(X_train)
X_test_std=processing.transform(X_test)

#cehcking existence of Multicollinearity in train set 
labels=['Fixed Acidity','Volatile Acidity','Citric Acid','Residual Sugar','Chlorides','Free Sulfur Dioxide','Total Sulfur Dioxide','Density','pH','Sulphates','Alcohol','Color']
variance_inflation_factor=[SMVIF(X_train_std,i) for i in range(X_train_std.shape[1])]
plt.figure(figsize=(22,8))
plt.bar(x=labels,height=variance_inflation_factor,width=0.5,tick_label=labels)
plt.title("Variance Inflation shown by Wine Features",fontsize=40)
plt.ylabel('VIF',fontsize=25)
plt.xlabel('Feature',fontsize=25)
plt.savefig("Variance_Inflation_Factor.png",format='png',dpi=1200)
plt.tight_layout()
plt.show()



# Dealing with mutlticollinearity applying Principle Component Analysis
U,s,V=np.linalg.svd(np.array(X_train_std),full_matrices=False)
eigen_vals=s**2/(len(X_train_std)-1)
eigen_vectors=np.transpose(V)
variance_explained=[100*i/sum(eigen_vals) for i in eigen_vals]
cummulative_variance=np.cumsum(variance_explained)
with plt.style.context('seaborn-whitegrid'):
    plt.figure(figsize=(10, 6))
    plt.bar(range(12), variance_explained, alpha=0.5, align='center',
            label='individual explained variance')
    plt.step(range(12), cummulative_variance, where='mid',
             label='cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc='best')
    plt.title("Variance explained by Principle Components",fontsize=30)
    plt.tight_layout()
plt.savefig('PCA_Variance_Explained.png', format='png', dpi=1200)
plt.show()

#transforming data in principle components. 10 prinpiple components are use as the explain 99 percent of the variance
X_train_std_transformed=U.dot(np.diag(s))[:,0:10]
X_test_std_transformed=X_test_std.dot(np.transpose(V))[:,0:10]


#estimating the parameters for Ridge Regression, Lasso Regression and Elastic Net through crossvalidation
def linear_model_parameter_selection(training_model,X_training_set,Y_training_set):
    trained_model=training_model.fit(X_training_set,Y_training_set)
    return trained_model

Ridge_CV_model=linear_model_parameter_selection(linear_model.RidgeCV(alphas=np.linspace(0,100,1000),scoring='neg_mean_squared_error',cv=10),X_train_std_transformed,Y_train)
ridge_alpha=Ridge_CV_model.alpha_

Lasso_CV_model=linear_model_parameter_selection(linear_model.LassoCV(cv=10,random_state=12,tol=0.00001),X_train_std_transformed,Y_train)
lasso_alpha=Lasso_CV_model.alpha_

Elastic_CV_model=linear_model_parameter_selection(linear_model.ElasticNetCV(l1_ratio=np.linspace(.0000000000001,10,50),eps=0.0001,n_alphas=1000,random_state=12,cv=10),X_train_std_transformed,Y_train)
elastic_model_alpha=Elastic_CV_model.alpha_
elastic_model_l1ratio=Elastic_CV_model.l1_ratio_


#prediction through different linear models using the parameters estimated earlier 
def Linear_Model(training_model,X_training_set,Y_training_set,X_test_set,Y_test_set):
    trained_model=training_model.fit(X_training_set,Y_training_set)
    Y_pred=trained_model.predict(X_test_set)
    model_mse=mean_squared_error(Y_test_set,Y_pred)
    model_r2_score=r2_score(Y_test_set,Y_pred)
    return (model_r2_score,model_mse,trained_model.coef_,Y_pred )

model_list=[linear_model.LinearRegression(), linear_model.Ridge(alpha=ridge_alpha,random_state=12),linear_model.Lasso(alpha=lasso_alpha,random_state=12),linear_model.ElasticNet(l1_ratio=elastic_model_l1ratio,alpha=elastic_model_alpha,random_state=12)] 
r2_score_list=[]
mse_score_list=[]
coef_list=[]    


for modl in model_list:
    X=Linear_Model(modl,X_train_std_transformed,Y_train,X_test_std_transformed,Y_test)
    r2_score_list.append(X[0])
    mse_score_list.append(X[1])
    coef_list.append(X[2])


#parameter selection and target prediction for different models, namely Random Forest Regressor, Logistic Regressor and Support Vector Regressor
def grid_search_method(pipeline,parameters,X_training_set,Y_training_set,X_test_set,Y_test_set):
    training_model=GridSearchCV(pipeline,parameters,cv=10)
    trained_model=training_model.fit(X_training_set,Y_training_set)
    Y_pred=trained_model.predict(X_test_set)
    model_mse=mean_squared_error(Y_test,Y_pred)
    model_r2_score=r2_score(Y_test_set,Y_pred)
    return (model_r2_score, model_mse,Y_pred,trained_model.best_estimator_)

RandomForest_pipeline = make_pipeline(RandomForestRegressor(n_estimators=100,random_state=12))
RandomForest_hyperparameters = { 'randomforestregressor__max_features' : ['auto', 'sqrt', 'log2'],'randomforestregressor__criterion':['mse'],'randomforestregressor__max_depth': [None, 5, 3, 1]}

Logistic_pipeline=make_pipeline(linear_model.LogisticRegression(random_state=12))
Logistic_hyperparameters={'logisticregression__penalty':['l1','l2'],'logisticregression__C':[0.1,0.5,1,2,5,10]}

SVR_pipeline=make_pipeline(svm.SVR(tol=0.00001))
SVR_hyperparameters={'svr__epsilon':[0.1,0.2,0.5,1],'svr__kernel':['rbf','poly','sigmoid'],'svr__C':[0.1,0.01,1,10]}


pipeline_list=[RandomForest_pipeline,Logistic_pipeline,SVR_pipeline]
hyperparameter_list=[RandomForest_hyperparameters,Logistic_hyperparameters,SVR_hyperparameters]

GridSearch_r2_score=[]
GridSearch_mse_score=[]
GridSearch_Y_pred=[]    
GridSearch_best_estimators=[]

for i in range(len(pipeline_list)):
    X=grid_search_method(pipeline_list[i],hyperparameter_list[i],X_train_std_transformed,Y_train,X_test_std_transformed,Y_test)
    GridSearch_r2_score.append(X[0])
    GridSearch_mse_score.append(X[1])
    GridSearch_Y_pred.append(X[2])    
    GridSearch_best_estimators.append(X[3])

#Performances of different models
Performance={'Model':['Linear Regression', 'Ridge Regression','Lasso Regression','Elastic Net','Random Forest','Logistic Regression','Support Vector'],
           'r2_score':[r2_score_list[0],r2_score_list[1],r2_score_list[2],r2_score_list[3],GridSearch_r2_score[0],GridSearch_r2_score[1],GridSearch_r2_score[2]],
           'mse_score':[mse_score_list[0],mse_score_list[1],mse_score_list[2],mse_score_list[3],GridSearch_mse_score[0],GridSearch_mse_score[1],GridSearch_mse_score[2]]}


Model_Performance = pd.DataFrame(Performance, columns = ['Model','r2_score','mse_score'])
Model_Performance.to_csv('Model_performance.csv')



