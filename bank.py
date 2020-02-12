import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.model_selection import train_test_split
#import keras
from keras.models import Sequential
from keras.layers import Dense
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
#By k-fold validation
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from datetime import datetime

now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Current Time End=", current_time)

df = pd.read_csv('bank-additional-full.csv',sep=';')
#df['default'] = df['default'].replace("unknown",np.nan)
#df.drop(['default'],axis=1,inplace=True)
df = df.replace("unknown",np.nan)
df['job'] = df['job'].replace(np.nan,df['job'].mode()[0])
df['marital'] = df['marital'].replace(np.nan,df['marital'].mode()[0])
df['education'] = df['education'].replace(np.nan,df['education'].mode()[0])
df['default'] = df['default'].replace(np.nan,df['default'].mode()[0])
df['housing'] = df['housing'].replace(np.nan,df['housing'].mode()[0])
df['loan'] = df['loan'].replace(np.nan,df['loan'].mode()[0])

#inputData = df.iloc[:,0:20].values
targetData = df.iloc[:,20].values

del df['y']

#columnTransformer = ColumnTransformer([('encoder',OneHotEncoder(),[1])],remainder = 'passthrough')
#inputData = np.array(columnTransformer.fit_transform(inputData),dtype = np.str)

inputData = pd.get_dummies(df,columns = ['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome'],drop_first=True)
#print('inputData',inputData)
#print('inputData head',pd.DataFrame(inputData).head())

inputData.to_csv('inputData.csv',index=False)
tempData = inputData

####################################################################################
#tempData = np.array(tempData, dtype = np.str)
#tempData = tempData[:, 1:]




inputData_train, inputData_test, targetData_train, targetData_test = train_test_split(tempData, targetData, test_size = 0.2, random_state = 0)

inputData_train = StandardScaler().fit_transform(inputData_train)
inputData_test = StandardScaler().fit_transform(inputData_test)
#print('tempData*********',tempData)
#print('targetData*********',targetData)
#print('inputData_train*********',inputData_train)
#print('inputData_test*********',inputData_test)

#####################################################################################

def build_classifier():
	#24=47+1/2
    sq = Sequential()
    sq.add(Dense(units = 24, kernel_initializer = 'uniform', activation = 'relu', input_dim = 47))
    sq.add(Dense(units = 24, kernel_initializer = 'uniform', activation = 'relu'))
    sq.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    sq.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return sq

classifier = KerasClassifier(build_classifier, batch_size = 10, epochs = 100)
accuracies = cross_val_score(estimator = classifier, X = inputData_train,  y = targetData_train, cv = 10 )

mean = accuracies.mean()
variance = accuracies.std()

now = datetime.now()
current_time = now.strftime("%H:%M:%S")
#confusion_matrix(y_true, y_pred)
print("Current Time End=", current_time)

#print('inputData',inputData)

#columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [1])], remainder='passthrough')
#X = np.array(columnTransformer.fit_transform(X), dtype = np.str)

#print(df.isnull().sum())
#print(df.head(5))

