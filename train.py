import numpy as np
import pandas as pd

#load dataset
def load_data():
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    print train.shape
    print test.shape

    submission = pd.DateFrame()
    submission["QuoteNumber"] = test["QuoteNumber"] 

    train = train.drop(['QuoteNumber','PropertyField6','GeographicField10A'],axis=1)
    train['Date'] = pd.to_datetime(pd.Series(train['Original_Quote_Date']))
    train['Year'] = train['Date'].apply(lambda x:int(str(x)[:4]))
    train['Month'] = train['Date'].apply(lambda x:int(str(x)[5:7]))
    train['Weekday'] = [train['Date'][i].dayofweek for i in range(len(train['Date']))]

    
    test = test.drop(['QuoteNumber','PropertyField6','GeographicField10A'],axis=1)
    test['Date'] = pd.to_datetime(pd.Series(test['Original_Quote_Date']))
    test['Year'] = test['Date'].apply(lambda x:int(str(x)[:4]))
    test['Month'] = test['Date'].apply(lambda x:int(str(x)[5:7]))
    test['Weekday'] = [test['Date'][i].dayofweek for i in range(len(test['Date']))]
    
    train = train.drop(['Date'],axis=1)
    test = test.drop(['Date'],axis=1)

    #fill na
    train = train.fillna(-1)
    test = train.fillna(-1)


    return train

dataset = load_data()
