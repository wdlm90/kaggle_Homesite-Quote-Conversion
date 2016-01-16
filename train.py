import numpy as np
import pandas as pd
from pandas import Series,DataFrame
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import log_loss,auc,roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
from sklearn.cross_validation import  train_test_split
import xgboost as xgb
#load dataset
print 'loading dataset...'
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
print 'finish loading dataset...'

train = train.drop(['QuoteNumber'],axis=1)

train['Date'] = pd.to_datetime(pd.Series(train['Original_Quote_Date']))
train['Year'] = train['Date'].dt.year
train['Month'] = train['Date'].dt.month
train['Weekday'] = train['Date'].dt.dayofweek

test['Date'] = pd.to_datetime(pd.Series(test['Original_Quote_Date']))
test['Year'] = test['Date'].dt.year
test['Month'] = test['Date'].dt.month
test['Weekday'] = test['Date'].dt.dayofweek

train.drop(['Date'],axis=1,inplace=True)
test.drop(['Date'],axis=1,inplace=True)
train.drop(['Original_Quote_Date'],axis=1,inplace=True)
test.drop(['Original_Quote_Date'],axis=1,inplace=True)
print 'feature shape:',np.shape(train)
#fill na
train.fillna(-1,inplace=True)
test.fillna(-1,inplace=True)
print 'preprocessing dataset...' 
#preprocess label
for f in train.columns:
    if train[f].dtype=='object':
        lbl=LabelEncoder()
        lbl.fit(np.unique(list(train[f].values)+list(test[f].values)))
        train[f]=lbl.transform(list(train[f].values))
        test[f]=lbl.transform(list(test[f].values))

train_sample = np.random.choice(train.index.values,150000)
train = train.ix[train_sample]
Y_train = train['QuoteConversion_Flag']
X_train = train.drop(['QuoteConversion_Flag'],axis=1)
X_test = test.drop(['QuoteNumber'],axis=1).copy()

#feature scale
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#feature selection

lsvc = LinearSVC(C=0.01,penalty="l1",dual=False).fit(X_train,Y_train)
model = SelectFromModel(lsvc,prefit=True)
X_train_new = model.transform(X_train)
X_test_new = model.transform(X_test)
print 'new feature shape:',np.shape(X_train_new)

# create classifier
clf_tree = ExtraTreesClassifier(n_estimators=580,max_features=np.shape(X_train_new)[1],criterion='entropy',min_samples_split=3,max_depth=30,min_samples_leaf=8)
clf_xgboost = xgb.XGBClassifier(n_estimators=25,nthread=1,max_depth=10,learning_rate=0.025,silent=True,subsample=0.8,colsample_bytree=0.8)
'''
#cross validation
print 'cross validation...'
scores_tree = cross_val_score(clf_tree,X_train_new,Y_train,cv=3,scoring='roc_auc')
print 'random tree classifier score is:%s' % scores_tree
scores_xgboost = cross_val_score(clf_xgboost,X_train_new,Y_train,cv=3,scoring='roc_auc')
print 'xgboost classifier score is:%s' % scores_xgboost
'''
#parameter optimization



#predict probability
print 'predict probability'
clf_tree.fit(X_train,Y_train)
pred = clf_tree.predict_proba(X_test)[:,1]

'''
#xgboost
print 'training model...'
params = {"objective":"binary:logistic"}
T_train_xgb = xgb.DMatrix(X_train,label=Y_train,missing=-999.0)
x_test_xgb = xgb.DMatrix(x_test,missing=-999.0)
gbm = xgb.train(params,T_train_xgb,20)
Y_pred = gbm.predict(X_test_xgb)
print 'finish training model...'
'''

#create submission
print 'create submission...'
submission =pd.DataFrame()
submission['QuoteNumber']=test['QuoteNumber']
submission['QuoteConversion_Flag']=pred
submission.to_csv('submission_tree_2015.1.16_3.csv',index=False)




