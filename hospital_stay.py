import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import QuantileTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier


from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics

np.set_printoptions(suppress=True)
import warnings
warnings.filterwarnings('ignore')

#https://www.kaggle.com/code/bigsmile99/length-of-stay-prediction-accuracy-99-71
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
from sklearn.preprocessing import LabelEncoder

train.head()

train.info()
train.Stay.unique()

# NA values in train dataset
train.isnull().sum().sort_values(ascending = False)

# NA values in test dataset
test.isnull().sum().sort_values(ascending = False)

# Dimension of train dataset
train.shape

# Dimension of test dataset
test.shape

# Number of distinct observations in train dataset 
for i in train.columns:
    print(i, ':', train[i].nunique())
    
# Number of distinct observations in test dataset
for i in test.columns:
    print(i, ':', test[i].nunique())    
    
#Replacing NA values in Bed Grade Column for both Train and Test datssets
train['Bed Grade'].fillna(train['Bed Grade'].mode()[0], inplace = True)
test['Bed Grade'].fillna(test['Bed Grade'].mode()[0], inplace = True)  

#Replacing NA values in  Column for both Train and Test datssets
train['City_Code_Patient'].fillna(train['City_Code_Patient'].mode()[0], inplace = True)
test['City_Code_Patient'].fillna(test['City_Code_Patient'].mode()[0], inplace = True)

# Label Encoding Stay column in train dataset
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train['Stay'] = le.fit_transform(train['Stay'].astype('str'))

train.head()

#Imputing dummy Stay column in test datset to concatenate with train dataset
test['Stay'] = -1
df = pd.concat([train, test])
df.shape

#Label Encoding all the columns in Train and test datasets
for i in ['Hospital_type_code','Hospital_region_code',  'Department',
          'Ward_Type', 'Ward_Facility_Code', 'Type of Admission', 'Severity of Illness', 'Age']:
    le = LabelEncoder()
    df[i] = le.fit_transform(df[i].astype(str))
    
#Spearating Train and Test Datasets
train = df[df['Stay']!=-1]
test = df[df['Stay']==-1]    
    
    
# Feature Engineering
#Label Encoding all the columns in Train and test datasets
def get_countid_enocde(train, test, cols, name):
  temp = train.groupby(cols)['case_id'].count().reset_index().rename(columns = {'case_id': name})
  temp2 = test.groupby(cols)['case_id'].count().reset_index().rename(columns = {'case_id': name})
  train = pd.merge(train, temp, how='left', on= cols)
  test = pd.merge(test,temp2, how='left', on= cols)
  train[name] = train[name].astype('float')
  test[name] = test[name].astype('float')
  train[name].fillna(np.median(temp[name]), inplace = True)
  test[name].fillna(np.median(temp2[name]), inplace = True)
  return train, test



# Droping duplicate columns
test1 = test.drop(['Stay', 'patientid','Ward_Facility_Code'], axis =1)
train1 = test.drop(['case_id',   'Ward_Facility_Code'], axis =1)

# Splitting train data for Naive Bayes and XGBoost
X1 = train1.drop('Stay', axis =1)
y1 = train1['Stay']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size =0.20, random_state =100)

from sklearn.naive_bayes import GaussianNB
target = y_train.values
features = X_train.values
classifier_nb = GaussianNB()
model_nb = classifier_nb.fit(features, target)

prediction_nb = model_nb.predict(X_test)
from sklearn.metrics import accuracy_score
acc_score_nb = accuracy_score(prediction_nb,y_test)
print("Acurracy of Naive Bayes :", acc_score_nb*100)



mdl = LinearRegression()
x_train, x_test, y_train, y_test = train_test_split(X1, y1, test_size = 0.30, random_state = 500)

mdl.fit( X1, y1 )
y_prediction = mdl.predict(x_test)
score=r2_score(y_test,y_prediction)
print("Accuarcy after applying multiple linear regression : - ")
print(score)
print( mean_squared_error(y_test,y_prediction))
print(np.sqrt(mean_squared_error(y_test,y_prediction)))
'''
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, y_train) * 100, 2)
acc_decision_tree

model = RandomForestRegressor()

#transforming target variable through quantile transformer
ttr = TransformedTargetRegressor(regressor=model, transformer=QuantileTransformer(output_distribution='normal'))
ttr.fit(X1,y1)
yhat = ttr.predict(x_test)
score2=r2_score(y_test,yhat)
print("accuracy of random forest :- ")
print(score2)
print(yhat)
r2_score(y_test, yhat), mean_absolute_error(y_test, yhat), np.sqrt(mean_squared_error(y_test, yhat))

# Train the model
ttr.fit(X_train, y_train)

# Make predictions on the training data
y_train_pred = ttr.predict(X_train)

# Make predictions on the test data
y_test_pred = ttr.predict(X_test)

print(X_train.shape)
print(y_train.shape)

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error

# Define a range of depths to test
depths = range(1, 21)  # Adjust the range as needed

# Initialize empty lists to store training and testing MSE values
train_mse = []
test_mse = []

# Loop through different depths and train Random Forest regressors
for depth in depths:
    model = RandomForestRegressor(max_depth=depth, random_state=100)
    model.fit(X_train, y_train)
    
    # Make predictions on training and test data
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate MSE for training and test data
    train_mse.append(mean_squared_error(y_train, y_train_pred))
    test_mse.append(mean_squared_error(y_test, y_test_pred))

# Plot the training and testing MSE
plt.figure(figsize=(10, 6))
plt.plot(depths, train_mse, label='Training MSE', marker='o')
plt.plot(depths, test_mse, label='Testing MSE', marker='o')
plt.title('Training vs. Testing MSE for Random Forest Regressor')
plt.xlabel('Max Depth')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.grid(True)
plt.show()

# Decision Tree

# Calculate MSE for training and test data
mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)

#  MSE_train is close to MSE_test, it suggests a good model fit without significant overfitting or underfitting.
print("Mean Squared Error (MSE) on Training Data:", mse_train)
print("Mean Squared Error (MSE) on Test Data:", mse_test)

'''







'''model = RandomForestClassifier()

X1, y1 = datasets.make_classification()

X_train, X_val, Y_train, Y_val = train_test_split(X1, y1,test_size = 0.15, random_state=222)

model.fit(X_train, Y_train)
    
print('Training Accuracy : ',
      metrics.accuracy_score(Y_train,
                             model.predict(X_train))*100)
print('Validation Accuracy : ',
      metrics.accuracy_score(Y_val,
                             model.predict(X_val))*100)'''

