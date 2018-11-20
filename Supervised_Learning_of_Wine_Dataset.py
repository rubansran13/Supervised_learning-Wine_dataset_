import pandas as pd
import numpy as np
from sklearn.svm import SVC
import seaborn as sns
from sklearn.model_selection import train_test_split as tts
from sklearn import preprocessing
from sklearn.linear_model import  LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score as AS
from statsmodels.stats.outliers_influence import variance_inflation_factor as SMVIF
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn.ensemble import RandomForestClassifier
from sklearn import naive_bayes as NB
from sklearn.metrics import precision_score,recall_score

#reading data from the file path
white_Wine=pd.read_csv("C:/Users/My Pc/Downloads/winequality-white.csv",sep=";")
red_Wine=pd.read_csv("C:/Users/My Pc/Downloads/winequality-red.csv",sep=";")


#combining the red and white data into one with one variable corresponding to the color of wine
white_Wine["color"]="W"
red_Wine["color"]="R"
wine=pd.concat([red_Wine,white_Wine], ignore_index=True)

#redefining columns names
wine.rename(columns={'fixed acidity': 'fixed_acidity','citric acid':'citric_acid','volatile acidity':'volatile_acidity','residual sugar':'residual_sugar','free sulfur dioxide':'free_sulfur_dioxide','total sulfur dioxide':'total_sulfur_dioxide'}, inplace=True)

#adding dummy variable for color
wine=pd.get_dummies(wine,columns=['color'])

wine.rename(columns={'fixed acidity': 'fixed_acidity','citric acid':'citric_acid','volatile acidity':'volatile_acidity','residual sugar':'residual_sugar','free sulfur dioxide':'free_sulfur_dioxide','total sulfur dioxide':'total_sulfur_dioxide'}, inplace=True)

#independent variables that will be used to predict quality of wine
features=['fixed_acidity','volatile_acidity','citric_acid','residual_sugar','chlorides','free_sulfur_dioxide','total_sulfur_dioxide','density','pH','sulphates','alcohol','color_R','color_W']

#separating independent and dependent variable
wine_features=wine[features]
wine_target=wine['quality']
wine_target[wine_target<6]=0
wine_target[wine_target==6]=1
wine_target[wine_target>6]=2


#checking the dataset contains any null values
wine.isnull().sum()

#train data and test data split  (75% train data-25% test data). Test data is kept for checking the performance of the models. 
X_train, X_test, Y_train,Y_test=tts(wine_features,wine_target,test_size=0.25,random_state=4,stratify =wine_target)
#removing mean and scaling to unit variance 
processing=preprocessing.StandardScaler()

# following line of code fits standard scaler on train (only on continuos variables) set along with transforming it and also concatenate categorical data to the standardized data.
X_train_std= np.concatenate((processing.fit_transform(X_train.iloc[:,:-2]),np.array(X_train.iloc[:,-2:]).reshape(len(X_train),2)),axis=1)

#standardizing test set and concatenate categorical data to standardized data 
X_test_std= np.concatenate((processing.transform(X_test.iloc[:,:-2]),np.array(X_test.iloc[:,-2:]).reshape(len(X_test),2)),axis=1)

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
labels=['Fixed Acidity','Volatile Acidity','Citric Acid','Residual Sugar','Chlorides','Free Sulfur Dioxide','Total Sulfur Dioxide','Density','pH','Sulphates','Alcohol']
VIF_data=X_train_std[:,0]
variance_inflation_factor=[SMVIF(X_train_std,i) for i in range(len(labels))]
plt.figure(figsize=(22,8))
plt.bar(x=labels,height=variance_inflation_factor,width=0.5,tick_label=labels)
plt.title("Variance Inflation shown by Wine Features",fontsize=40)
plt.ylabel('VIF',fontsize=25)
plt.xlabel('Feature',fontsize=25)
plt.savefig("Variance_Inflation_Factor.png",format='png',dpi=1200)
plt.tight_layout()
plt.show()



# Dealing with mutlticollinearity applying Principle Component Analysis
U,s,V=np.linalg.svd(np.array(X_train_std[:,0:11]),full_matrices=False)
eigen_vals=s**2/(len(X_train_std[:,::-2])-1)
eigen_vectors=np.transpose(V)
eigen_val_sum=sum(eigen_vals)
variance_explained=[(100*i/eigen_val_sum) for i in eigen_vals]
cummulative_variance=np.cumsum(variance_explained)
with plt.style.context('seaborn-whitegrid'):
    plt.figure(figsize=(10, 6))
    plt.bar(range(X_train_std[:,:-2].shape[1]), variance_explained, alpha=0.5, align='center',
            label='individual explained variance')
    plt.step(range(X_train_std[:,:-2].shape[1]), cummulative_variance, where='mid',
             label='cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc='best')
    plt.title("Variance explained by Principle Components",fontsize=30)
    plt.tight_layout()
plt.savefig('PCA_Variance_Explained.png', format='png', dpi=1200)
plt.show()

#transforming data in principle components. 10 prinpiple components are used as the explain 99 percent of the variance
X_train_std_transformed=np.concatenate((U.dot(np.diag(s))[:,0:10],np.array(X_train.iloc[:,-2:]).reshape(len(X_train),2)),axis=1)
X_test_std_transformed=np.concatenate((X_test_std[:,:-2].dot(np.transpose(V))[:,0:10],np.array(X_test.iloc[:,-2:]).reshape(len(X_test),2)),axis=1)


"""Using Classification techniques for wine quality analysis"""


clf=NB.GaussianNB()
clf.fit(X_train_std_transformed, Y_train)
Y_pred_GNB = clf.predict(X_test_std_transformed)
train_accuracy_score_GNB=AS(Y_train,clf.predict(X_train_std_transformed))
test_accuracy_score_GNB=AS(Y_test,Y_pred_GNB)
confusion_matrix_GNB=confusion_matrix(Y_test, Y_pred_GNB)
precision_score_GNB=precision_score(Y_test,Y_pred_GNB,average='weighted')
recall_score_GNB=recall_score(Y_test,Y_pred_GNB,average='weighted')

#parameter selection and prediction for classification models
def Classification_grid_search_method(pipeline,parameters,X_training_set,Y_training_set,X_test_set,Y_test_set):
    training_model=GridSearchCV(pipeline,parameters,cv=10,scoring='accuracy')
    trained_model=training_model.fit(X_training_set,Y_training_set)
    Y_pred=trained_model.predict(X_test_set)
    test_Accuracy_Score=AS(Y_test_set,Y_pred)
    train_Accuracy_Score=AS(Y_training_set,trained_model.predict(X_training_set))
    Confusion_Matrix=confusion_matrix(Y_test,Y_pred)
    return(Y_pred,train_Accuracy_Score,test_Accuracy_Score,Confusion_Matrix,trained_model.best_estimator_)


Logistic_pipeline=make_pipeline(LogisticRegression(random_state=12))
Logistic_hyperparameters={'logisticregression__penalty':['l1','l2'],'logisticregression__C':[0.1,0.5,1,2,5,10]}

DTC_pipeline=make_pipeline(DTC(random_state=4,))
DTC_hyperparameters={'decisiontreeclassifier__criterion':['gini','entropy'],'decisiontreeclassifier__splitter':['best','random'],'decisiontreeclassifier__max_depth':[5,8,12,15,18,20,22,25],'decisiontreeclassifier__min_samples_split':[2,10,20,30],'decisiontreeclassifier__max_features':['auto','sqrt',None],}

Nearest_Neighbors_pipeline=make_pipeline(KNC())
Nearest_Neighbors_hyperparameters={'kneighborsclassifier__n_neighbors':[2,5,10,20,30,40,50],'kneighborsclassifier__weights':['distance','uniform'],'kneighborsclassifier__algorithm' : ['auto', 'ball_tree', 'kd_tree', 'brute'],'kneighborsclassifier__leaf_size':[10,20,30,40,50,75,100],'kneighborsclassifier__p':[1,2]}


RFC_pipeline=make_pipeline(RandomForestClassifier(warm_start=True,random_state=4))
RFC_hyperparameters={'randomforestclassifier__n_estimators':[10,50,100],'randomforestclassifier__criterion':['gini','entropy'],'randomforestclassifier__max_depth':[None,20,50,100],'randomforestclassifier__max_features':[None,'sqrt'],'randomforestclassifier__min_samples_split':[2,10,20],'randomforestclassifier__min_samples_leaf':[1,5,10,20]}

SVC_pipeline=make_pipeline(SVC(random_state=4))
SVC_hyperparameters={'svc__C':[0.1,1,10,20,50],'svc__degree':[1,2,3,4,5],'svc__kernel':['rbf','poly','sigmoid']}


Classification_pipeline=[Logistic_pipeline,DTC_pipeline,Nearest_Neighbors_pipeline,RFC_pipeline,SVC_pipeline]
Classification_hyperparameters=[Logistic_hyperparameters,DTC_hyperparameters,Nearest_Neighbors_hyperparameters,RFC_hyperparameters,SVC_hyperparameters]



Classification_Predicted_Target=[]    
Classification_train_Accuracy=[]
Classification_test_Accuracy=[] 
Classification_Confusion_matrix=[] 
Classification_Best_Estimators=[]


for i in range(len(Classification_pipeline)):
    X=Classification_grid_search_method(Classification_pipeline[i],Classification_hyperparameters[i],X_train_std_transformed,Y_train,X_test_std_transformed,Y_test)
    Classification_Predicted_Target.append(X[0])      
    Classification_train_Accuracy.append(X[1]) 
    Classification_test_Accuracy.append(X[2]) 
    Classification_Confusion_matrix.append(X[3])  
    Classification_Best_Estimators.append(X[4])
    


Classification_Performance={"Model":['Gaussian Naive Bayes','Logistic Regression','Decision Tree Classifier','K Neighbors Classifier','Random Forest Classifier','Support Vector Classifier'],
                            'Train Accuracy Score':[train_accuracy_score_GNB,Classification_train_Accuracy[0],Classification_train_Accuracy[1],Classification_train_Accuracy[2],Classification_train_Accuracy[3],Classification_train_Accuracy[4]],
                            'Test Accuracy Score':[test_accuracy_score_GNB,Classification_test_Accuracy[0],Classification_test_Accuracy[1],Classification_test_Accuracy[2],Classification_test_Accuracy[3],Classification_test_Accuracy[4]]}
Classification_Model_Performance = pd.DataFrame(Classification_Performance, columns = ['Model','Train Accuracy Score','Test Accuracy Score'])
Classification_Model_Performance.to_csv('Classification_Model_Performance.csv')


