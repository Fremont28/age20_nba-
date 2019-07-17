import xgboost as xgb
from sklearn.metrics import mean_squared_error
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import GradientBoostingRegressor

#import data 
nba_yr=pd.read_csv("nba_todos.csv")

#subset by age season 
age_19=nba_yr[nba_yr.Age==19] 
age_20=nba_yr[nba_yr.Age==20] 
age_21=nba_yr[nba_yr.Age==21]
age_22=nba_yr[nba_yr.Age==22] 
age_23=nba_yr[nba_yr.Age==23]
age_24=nba_yr[nba_yr.Age==24] 
age_25=nba_yr[nba_yr.Age==25]
age_26=nba_yr[nba_yr.Age==26] 
age_27=nba_yr[nba_yr.Age==27]
age_28=nba_yr[nba_yr.Age==28] 
age_29=nba_yr[nba_yr.Age==29]
age_30=nba_yr[nba_yr.Age==30] 
age_31=nba_yr[nba_yr.Age==31]
age_32=nba_yr[nba_yr.Age==32] 
age_33=nba_yr[nba_yr.Age==33]
age_34=nba_yr[nba_yr.Age==34] 
age_35=nba_yr[nba_yr.Age==35]
age_36=nba_yr[nba_yr.Age==36] 
age_37=nba_yr[nba_yr.Age==37]
age_38=nba_yr[nba_yr.Age==38] 
age_39=nba_yr[nba_yr.Age==39]
age_40=nba_yr[nba_yr.Age==40] 
age_41=nba_yr[nba_yr.Age==41]
age_42=nba_yr[nba_yr.Age==42] 

#age 20 season (00-01' to 18-19')
age_20_train=nba_yr[nba_yr.Age==20]
age_20_train=age_20_train[["Player","MP","G","GS","FG","FGA","2P","2PA","3P","3PA","FT","FTA","ORB","DRB","AST","STL","BLK","TOV","PTS","PF","FG%","2P%","3P%","FT%"]]
age_20=nba_yr[nba_yr.Age==21]
age_20_trainY=age_20[["Player","VORP"]] 

age_20_trainY=pd.DataFrame(age_20_trainY)
df=age_20_train
df.reset_index(level=0,inplace=True)
labels=age_20_trainY 
labels.reset_index(level=0,inplace=True) 

old=pd.merge(df,labels,on="Player")
df1=old.iloc[:,1:25]
labels=old['VORP']


#split into train and test sets 
X_train,X_test,y_train,y_test=train_test_split(df1,labels,test_size=0.30,random_state=34)
X_train.head(3)
X_train1=X_train.iloc[:,1:X_train.shape[0]]
X_test1=X_test.iloc[:,1:X_test.shape[0]]
X_train1=X_train1.fillna(X_train1.median())
X_test1=X_test1.fillna(X_test1.median())
y_train=pd.DataFrame(y_train)
y_test=pd.DataFrame(y_test)

#baseline metrics 
#naive baseline is the median
median_pred =y_train['VORP'].median()
median_preds = [median_pred for _ in range(len(y_test))]
true = y_test['VORP']

base_mae=np.mean(abs(median_preds-true))
base_rmse=np.sqrt(np.mean((median_preds-true)**2))

#models----- 
#i. Linear Regression  
mod1=LinearRegression()
mod1.fit(X_train1,y_train)
preds=mod1.predict(X_test1)

mae=np.mean(abs(preds-y_test)) #0.83 
rmse=np.sqrt(np.mean((preds-y_test)**2)) #1.02 

#ii. elastic net 
mod2=ElasticNet(alpha=1.0,l1_ratio=0.5)
mod2.fit(X_train1,y_train)
preds=mod2.predict(X_test1)
preds=preds.reshape(65,1)

mae=np.mean(abs(preds-y_test)) #0.78 
rmse=np.sqrt(np.mean((preds-y_test)**2)) #1.07 

#iii. gradient boosting regressor 
mod3=GradientBoostingRegressor()
mod3.fit(X_train1,y_train)
preds=mod3.predict(X_test1)
preds=preds.reshape(65,1)

mae=np.mean(abs(preds-y_test)) #0.74  
rmse=np.sqrt(np.mean((preds-y_test)**2)) #1.0 

preds=pd.DataFrame(preds)
preds.columns=['pred_vorp']
actual=pd.DataFrame(y_test)
actual.reset_index(level=0,inplace=True)
players=X_test['Player']
players=pd.DataFrame(players)
players.reset_index(level=0,inplace=True)
comb=pd.concat([players,actual,preds],axis=1)
comb 

#this year's 20 year olds 
now=nba_yr[nba_yr.Season =="2018-19"]
now['Age']=now['Age']+1 
now_20=now[now.Age==20]

#train
old_train=pd.concat([X_train1,X_test1],axis=0)
old_labels=pd.concat([y_train,y_test],axis=0)
old_labels=old_labels.iloc[:,0:1]

#test 
now_test=now_20[["MP","G","GS","FG","FGA","2P","2PA","3P","3PA","FT","FTA","ORB","DRB","AST","STL","BLK","TOV","PTS","PF","FG%","2P%","3P%","FT%"]]
now_test.shape[1]

#gradient boost model 
modGB=GradientBoostingRegressor()
modGB.fit(old_train,old_labels)
preds=modGB.predict(now_test)
preds=pd.DataFrame(preds)
preds.columns=['Pred_VORP']
player=now_20['Player']
player=pd.DataFrame(player) 
player.reset_index(level=0,inplace=True)
now20_preds=pd.concat([player,preds],axis=1)
now20_preds.sort_values(by="Pred_VORP")
now20_preds.to_csv("now20_preds.csv")


