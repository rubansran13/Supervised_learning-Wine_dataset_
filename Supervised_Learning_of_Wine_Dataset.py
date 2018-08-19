import pandas as pd
import numpy as np
from sklearn import svm
import seaborn as sns
from sklearn.model_selection import train_test_split as tts
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor 
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score,confusion_matrix
from sklearn import linear_model
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score as AS
from statsmodels.stats.outliers_influence import variance_inflation_factor as SMVIF
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn.ensemble import RandomForestClassifier
from sklearn import naive_bayes as NB
from sklearn.metrics import precision_score,recall_score

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

#checking the dataset contains any null values
wine.isnull().sum()

#train data and test data split  (75% train data-25% test data). Test data is kept for checking the performance of the models. 
X_train, X_test, Y_train,Y_test=tts(wine_features,wine_target,test_size=0.25,random_state=4,stratify =wine_target)

#removing mean and scaling to unit variance 
processing=preprocessing.StandardScaler()
X_train_std= processing.fit_transform(X_train)
X_test_std=processing.transform(X_test)

#looking in the train set
X_train.describe()

#Histogram of Targeet Variable
sns.countplot(Y_train)

#indendent variable correlation plot
correlation_wine_features=X_train.corr()
axes = plt.axes()
plt.subplots(figsize=(10,10))
axes = plt.axes()
axes.set_title("Wine Characteristic Correlation Heatmap ")
sns.heatmap(correlation_wine_features, 
            xticklabels=correlation_wine_features.columns.values,
            yticklabels=correlation_wine_features.columns.values,cmap="Blues")
plt.savefig("Wine Characteristic Correlation Heatmap(Train set).png",format='png',dpi=1200)

#corelation has shown higher values for some features thus requiring furtger analysis

#cehcking existence of Multicollinearity in train set through Variance Inflation Analysis
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
eigen_val_sum=0
for i in eigen_vals:
    eigen_val_sum+=1
variance_explained=[(100*i/eigen_val_sum) for i in eigen_vals]
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

#transforming data in principle components. 10 prinpiple components are used as the explain 99 percent of the variance
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
    Y_pred_class=round(pd.Series(Y_pred))
    Accuracy_Score=AS(Y_test_set,Y_pred_class)
    Confusion_Matrix=confusion_matrix(Y_test,Y_pred_class)
    model_mse=mean_squared_error(Y_test_set,Y_pred)
    model_r2_score=r2_score(Y_test_set,Y_pred)
    minimum_predicted=min(Y_pred)
    maximum_predicted=max(Y_pred)
    return (model_r2_score,model_mse,trained_model.coef_,Y_pred,Y_pred_class,Accuracy_Score,Confusion_Matrix,minimum_predicted,maximum_predicted )

model_list=[linear_model.LinearRegression(), linear_model.Ridge(alpha=ridge_alpha,random_state=12),linear_model.Lasso(alpha=lasso_alpha,random_state=12),linear_model.ElasticNet(l1_ratio=elastic_model_l1ratio,alpha=elastic_model_alpha,random_state=12)] 
r2_score_list=[]
mse_score_list=[]
coef_list=[]  
Predicted_Target=[]  
Target_Class=[]
Accuracy_Score_list=[]
Confusion_matrix=[]
Minimum_of_predicted_target=[]
Maximum_of_predicted_Target=[]
for modl in model_list:
    X=Linear_Model(modl,X_train_std_transformed,Y_train,X_test_std_transformed,Y_test)
    r2_score_list.append(X[0])
    mse_score_list.append(X[1])
    coef_list.append(X[2])
    Predicted_Target.append(X[3]) 
    Target_Class.append(X[4])
    Accuracy_Score_list.append(X[5])
    Confusion_matrix.append(X[6])
    Minimum_of_predicted_target.append(X[7])
    Maximum_of_predicted_Target.append(X[8])

#parameter selection and target prediction for different models, namely Random Forest Regressor, Logistic Regressor and Support Vector Regressor
def grid_search_method(pipeline,parameters,X_training_set,Y_training_set,X_test_set,Y_test_set):
    training_model=GridSearchCV(pipeline,parameters,cv=10)
    trained_model=training_model.fit(X_training_set,Y_training_set)
    Y_pred=trained_model.predict(X_test_set)
    model_mse=mean_squared_error(Y_test,Y_pred)
    model_r2_score=r2_score(Y_test_set,Y_pred)
    Y_pred_class=round(pd.Series(Y_pred))
    Accuracy_Score=AS(Y_test_set,Y_pred_class)
    Confusion_Matrix=confusion_matrix(Y_test,Y_pred_class)
    minimum_predicted=min(Y_pred)
    maximum_predicted=max(Y_pred)
    return (model_r2_score, model_mse,Y_pred,trained_model.best_estimator_,Y_pred_class,Accuracy_Score,Confusion_Matrix,minimum_predicted,maximum_predicted)

RandomForest_pipeline = make_pipeline(RandomForestRegressor(n_estimators=100,random_state=12))
RandomForest_hyperparameters = { 'randomforestregressor__max_features' : ['auto', 'sqrt', 'log2'],'randomforestregressor__criterion':['mse'],'randomforestregressor__max_depth': [None, 5, 3, 1]}

Logistic_pipeline=make_pipeline(linear_model.LogisticRegression(random_state=12))
Logistic_hyperparameters={'logisticregression__penalty':['l1','l2'],'logisticregression__C':[0.1,0.5,1,2,5,10]}

SVR_pipeline=make_pipeline(svm.SVR(tol=0.00001))
SVR_hyperparameters={'svr__epsilon':[0.1,0.2,0.5,1],'svr__kernel':['rbf','poly','sigmoid'],'svr__C':[0.1,0.01,1,10]}

pipeline_list=[RandomForest_pipeline,Logistic_pipeline,SVR_pipeline]
hyperparameter_list=[RandomForest_hyperparameters,Logistic_hyperparameters,SVR_hyperparameters]

GridSearch_r2_score=[]
GridSearch_MSE_score=[]
GridSearch_Predicted_Target=[]    
GridSearch_Best_Estimators=[]  
GridSearch_Target_Class=[]
GridSearch_Accuracy_Score=[]
GridSearch_Confusion_matrix=[]
GridSearch_Minimum_of_predicted_target=[]
GridSearch_Maximum_of_predicted_Target=[]


for i in range(len(pipeline_list)):
    X=grid_search_method(pipeline_list[i],hyperparameter_list[i],X_train_std_transformed,Y_train,X_test_std_transformed,Y_test)
    GridSearch_r2_score.append(X[0])
    GridSearch_MSE_score.append(X[1])
    GridSearch_Predicted_Target.append(X[2])    
    GridSearch_Best_Estimators.append(X[3])  
    GridSearch_Target_Class.append(X[4])  
    GridSearch_Accuracy_Score.append(X[5])  
    GridSearch_Confusion_matrix.append(X[6])  
    GridSearch_Minimum_of_predicted_target.append(X[7])  
    GridSearch_Maximum_of_predicted_Target.append(X[8])  

#Performances of different models
Regression_Performance={'Model':['Linear Regression', 'Ridge Regression','Lasso Regression','Elastic Net','Random Forest','Logistic Regression','Support Vector'],
           'r2_Score':[r2_score_list[0],r2_score_list[1],r2_score_list[2],r2_score_list[3],GridSearch_r2_score[0],GridSearch_r2_score[1],GridSearch_r2_score[2]],
           'MSE_Score':[mse_score_list[0],mse_score_list[1],mse_score_list[2],mse_score_list[3],GridSearch_MSE_score[0],GridSearch_MSE_score[1],GridSearch_MSE_score[2]],
           'Accuracy_Score':[Accuracy_Score_list[0],Accuracy_Score_list[1],Accuracy_Score_list[2],Accuracy_Score_list[3],GridSearch_Accuracy_Score[0],GridSearch_Accuracy_Score[1],GridSearch_Accuracy_Score[2]]}


Regression_Model_Performance = pd.DataFrame(Regression_Performance, columns = ['Model','r2_Score','MSE_Score','Accuracy_Score'])
Regression_Model_Performance.to_csv('Regression_Model_Performance.csv')




"""Using Classification techniques for wine quality analysis"""


clf=NB.GaussianNB()
clf.fit(X_train_std_transformed, Y_train)
Y_pred_GNB = clf.predict(X_test_std_transformed)
accuracy_score_GNB=AS(Y_test,Y_pred_GNB)
confusion_matrix_GNB=confusion_matrix(Y_test, Y_pred_GNB)
precision_score_GNB=precision_score(Y_test,Y_pred_GNB,average='weighted')
recall_score_GNB=recall_score(Y_test,Y_pred_GNB,average='weighted')

#parameter selection and prediction for classification models
def Classification_grid_search_method(pipeline,parameters,X_training_set,Y_training_set,X_test_set,Y_test_set):
    training_model=GridSearchCV(pipeline,parameters,cv=10,scoring='accuracy')
    trained_model=training_model.fit(X_training_set,Y_training_set)
    Y_pred=trained_model.predict(X_test_set)
    Accuracy_Score=AS(Y_test_set,Y_pred)
    Confusion_Matrix=confusion_matrix(Y_test,Y_pred)
    Precision=precision_score(Y_test,Y_pred,average='weighted')
    Recall=recall_score(Y_test,Y_pred,average='weighted')
    return(Precision,Recall,Y_pred,Accuracy_Score,Confusion_Matrix,trained_model.best_estimator_)


DTC_pipeline=make_pipeline(DTC(random_state=4,))
DTC_hyperparameters={'decisiontreeclassifier__criterion':['gini','entropy'],'decisiontreeclassifier__splitter':['best','random'],'decisiontreeclassifier__max_depth':[None,50,75,100],'decisiontreeclassifier__min_samples_split':[2,10,20,30],'decisiontreeclassifier__max_features':['auto','sqrt',None],}

Nearest_Neighbors_pipeline=make_pipeline(KNC())
Nearest_Neighbors_hyperparameters={'kneighborsclassifier__n_neighbors':[2,5,10,20,30,40,50],'kneighborsclassifier__weights':['distance','uniform'],'kneighborsclassifier__algorithm' : ['auto', 'ball_tree', 'kd_tree', 'brute'],'kneighborsclassifier__leaf_size':[10,20,30,40,50,75,100],'kneighborsclassifier__p':[1,2]}


RFC_pipeline=make_pipeline(RandomForestClassifier(warm_start=True,random_state=4))
RFC_hyperparameters={'randomforestclassifier__n_estimators':[10,50,100],'randomforestclassifier__criterion':['gini','entropy'],'randomforestclassifier__max_depth':[None,20,50,100],'randomforestclassifier__max_features':[None,'sqrt'],'randomforestclassifier__min_samples_split':[2,10,20],'randomforestclassifier__min_samples_leaf':[1,5,10,20]}

Classification_pipeline=[DTC_pipeline,Nearest_Neighbors_pipeline,RFC_pipeline]
Classification_hyperparameters=[DTC_hyperparameters,Nearest_Neighbors_hyperparameters,RFC_hyperparameters]

Classification_Predicted_Target=[]    
Classification_Accuracy=[] 
Classification_Confusion_matrix=[] 
Classification_Best_Estimators=[]
Classification_Precision=[]
Classification_Recall=[]
for i in range(len(Classification_pipeline)):
    X=Classification_grid_search_method(Classification_pipeline[i],Classification_hyperparameters[i],X_train_std_transformed,Y_train,X_test_std_transformed,Y_test)
    Classification_Predicted_Target.append(X[2])      
    Classification_Accuracy.append(X[3])  
    Classification_Confusion_matrix.append(X[4])  
    Classification_Best_Estimators.append(X[5])
    Classification_Precision.append(X[0])
    Classification_Recall.append(X[1])


Classification_Performance={"model":['Gaussian Naive Bayes','Decision Tree Classifier','K Neighbors Classifier','Random Forest Classifier'],
                            'Accuracy Score':[accuracy_score_GNB,Classification_Accuracy[0],Classification_Accuracy[1],Classification_Accuracy[2]],
                            'Precision Score':[precision_score_GNB,Classification_Precision[0],Classification_Precision[1],Classification_Precision[2]],
                            'Recall Score':[recall_score_GNB,Classification_Recall[0],Classification_Recall[1],Classification_Recall[2]]}

Classification_Model_Performance = pd.DataFrame(Classification_Performance, columns = ['model','Accuracy Score','Precision Score', 'Recall Score'])
Classification_Model_Performance.to_csv('Classification_Model_Performance.csv')
