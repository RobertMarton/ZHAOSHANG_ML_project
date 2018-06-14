#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import lightgbm as lgb
from datetime import datetime

model_path = 'model'
prediction_path = 'prediction'
feat_path = 'feat'
version = datetime.now().strftime("%m%d%H%M")
print('version:',version)



file_train_agg = 'train/train_agg.csv'
file_train_log = 'train/train_log.csv'
file_train_flg = 'train/train_flg.csv'

file_test_agg = 'test/test_agg.csv'
file_test_log = 'test/test_log.csv'



def feat_df_agg():
    df1_agg = pd.read_table(file_train_agg).sort_values(['USRID'],ascending=[1]).reset_index(drop=True)
    df2_agg = pd.read_table(file_test_agg).sort_values(['USRID'],ascending=[1]).reset_index(drop=True)
    
    train_key = df1_agg[['USRID']]
    test_key = df2_agg[['USRID']]
    
    df_agg = pd.concat([df1_agg,df2_agg]).sort_values(['USRID'],ascending=[1]).reset_index(drop=True)
    
    categories = ['V2','V4','V5']
    
    df_agg.to_csv(feat_path+'/df_agg.csv',index=False)
    train_key.to_csv(feat_path+'/train_key.csv',index=False)
    test_key.to_csv(feat_path+'/test_key.csv',index=False)
    pd.DataFrame({'categories':categories}).to_csv(feat_path+'/df_agg_categories.csv',index=False)
    

def get_feat_df_agg():
    df = pd.read_csv(feat_path+'/df_agg.csv')
    categories = list(pd.read_csv(feat_path+'/df_agg_categories.csv')['categories'])
    return df,categories
    

def get_train_test_key():
    train_key = pd.read_csv(feat_path+'/train_key.csv')
    test_key = pd.read_csv(feat_path+'/test_key.csv')
    return train_key,test_key


def feat_df_log():
    df1_log = pd.read_table(file_train_log,parse_dates=['OCC_TIM'])
    df2_log = pd.read_table(file_test_log,parse_dates=['OCC_TIM'])
    
    df_log = pd.concat([df1_log,df2_log])
    df_log = df_log.sort_values(['USRID','OCC_TIM','EVT_LBL'],ascending=[1,1,0]).reset_index(drop=True)
    
    EVT_LBL = df_log['EVT_LBL'].str.split('-',expand=True)
    EVT_LBL.columns = ['EVT_LBL_0','EVT_LBL_1','EVT_LBL_2']
    df_log = pd.concat([df_log,EVT_LBL[['EVT_LBL_0','EVT_LBL_1']]],axis=1)
    for col in ['EVT_LBL','EVT_LBL_0','EVT_LBL_1']:
        df_log[col] = pd.Categorical(df_log[col]).labels
    
    df_log.to_csv(feat_path+'/df_log.csv',index=False)
    

def get_feat_df_log():
    return pd.read_csv(feat_path+'/df_log.csv',parse_dates=['OCC_TIM'])

# 最后10次的点击
def feat_log_1():
    N = 30
    df_log = get_feat_df_log()
    categories = []
    tmp = df_log.groupby('USRID').tail(N)
    tmp['time_to_4_1'] = ((datetime(2018,4,1,0,0,0) - tmp['OCC_TIM']).astype('int64'))//1000000000
    tmp['rank'] = tmp.groupby('USRID')['time_to_4_1'].rank(method='first').astype('int')
    merges = []
    for col in ['EVT_LBL','EVT_LBL_0','EVT_LBL_1','time_to_4_1']:
        pvt = tmp.pivot('USRID','rank',col)
        pvt.columns = ['{}_last{}'.format(col,i) for i in pvt.columns]
        
        if col in ['EVT_LBL','EVT_LBL_0','EVT_LBL_1']:
            categories.extend(pvt.columns.values)
        pvt = pvt.reset_index()
        merges.append(pvt)
    df = merges[0]
    for df_merge in merges[1:]:
        df = df.merge(df_merge,on='USRID',how='left')
    
    pd.DataFrame({'categories':categories}).to_csv(feat_path+'/feat_log_1_categories.csv',index=False)
    df.to_csv(feat_path+'/feat_log_1.csv',index=False)
    
def get_feat_log_1():
    df = pd.read_csv(feat_path+'/feat_log_1.csv')
    categories = list(pd.read_csv(feat_path+'/feat_log_1_categories.csv')['categories'])
    return df,categories
    
def modeling(X,Y,categorical):
    seed = 0
    EARLY_STOP = 100
    OPT_ROUNDS =0
    MAX_ROUNDS = 3000
    VALIDATE = True
    print(X.info())
    params = {
        'boosting_type': 'gbdt',  
        # 'drop_rate': 0.09,
        'metric': 'auc',
        'objective': 'binary',
        'learning_rate': 0.01,
        'num_leaves': 31,  
        'max_depth': -1,
        'min_child_samples': 20,
        'max_bin': 255,
        'subsample': 1,
        'subsample_freq': 0,
        'colsample_bytree': 0.8,
        'min_child_weight': 0.001,
        'subsample_for_bin': 200000,
        'min_split_gain': 0,
        'reg_alpha': 0,
        'reg_lambda': 0,
        'scale_pos_weight':2,
    }
    
    
    if VALIDATE:
        print("Start train and validate...")
    
        dtrain = lgb.Dataset(X, 
                         label=Y,
                         feature_name=list(X.columns),
                         categorical_feature=categorical)   
     
        
        eval_hist = lgb.cv(params, 
                      dtrain, 
                      nfold = 4,
                      
                      num_boost_round=MAX_ROUNDS,
                      early_stopping_rounds=EARLY_STOP,
                      verbose_eval=10, 
                      seed = seed,)
    
        OPT_ROUNDS = len(eval_hist['auc-mean'])
        print('OPT_ROUNDS:',OPT_ROUNDS)
    dtrain = lgb.Dataset(X, 
                         label=Y,
                         feature_name=list(X.columns),
                         categorical_feature=categorical)   
     
    
    model = lgb.train(params,dtrain,num_boost_round=OPT_ROUNDS)
    
    importances = pd.DataFrame({'features':model.feature_name(),
                 'importances':model.feature_importance()})
    
    importances.sort_values('importances',ascending=False,inplace=True)
    
    
    model.save_model(model_path+'/{}.model'.format(version))
    importances.to_csv(model_path+'/{}_mportances.csv'.format(version),index=False)
    
    return model

def predict(model,test):
    Y = model.predict(test[model.feature_name()])
    pd.DataFrame({'USRID':test['USRID'],'RST':Y},columns=['USRID','RST']).to_csv(prediction_path+'/{}.csv'.format(version),sep='\t',index=False)
    

def train_and_predict():
    train,test,categorical = get_features()
    
    flg = pd.read_table(file_train_flg)
    train = train.merge(flg,on=['USRID'],how='left')
    
    X = train.drop(['USRID','FLAG'],axis=1)
    Y = train['FLAG']
    model = modeling(X,Y,categorical)
    
    predict(model,test)
    

def feature_engineering():
    feat_df_agg()
    feat_df_log()
    feat_log_1()
    

def feature_update():
    feat_df_agg()


def train_test_merge_df(train,test,df):
    train = train.merge(df,on='USRID',how='left')
    test = test.merge(df,on='USRID',how='left')
    return train,test

def get_features():
    categorical = []
    train,test = get_train_test_key()
    
    df,categories = get_feat_df_agg()
    categorical.extend(categories)
    train,test = train_test_merge_df(train,test,df)
    
    df,categories = get_feat_log_1()
    categorical.extend(categories)
    train,test = train_test_merge_df(train,test,df)
    
    return train,test,categorical



if __name__=='__main__':
    # feature_engineering()
    # feature_update()

    train_and_predict()
    


