#Feature Selection RFE 
import pandas as pd
import numpy as np
import string
import csv
import sys
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from decimal import Decimal
url = 'Cleaned_merge_new1.csv'
df_clean = pd.read_csv(url)
df_clean.shape

df_clean.columns[df_clean.isnull().any()].tolist()
name=['Sex','Breed','Color','Age','RaiserState','GoodAppetite','Health','StoolFirm','EnergyLevel','EliminationInCrate',
'QuietInCrate',
'RespondsToCommandKennel','BarksExcessively','NoInappropriateChewing','Housemanners','LeftUnattended',
'EliminationInHouse','PlaybitePeople','StealsFood','OnFurniture','RaidsGarbage','CounterSurfingJumpOnDoors',
'JumpOnPeople','FriendlyWAnimals','GoodWKids','GoodWStrangers','WalksWellOnLeash','KnowCommandGetBusy','EliminatesOnRoute',
'ChasingAnimals','TrafficFear','NoiseFear','Stairs','SitsOnCommand','DownOnCommand','StaysOnCommand',
'ComeOnLeash','ComeOffLeash','CanGivePills','EarCleaning','NailCutting','AttendsClasses','BehavesWellClass','AttendsHomeSwitches',
'StatusCode']
array = df_clean.values
X1 = array[:,0:43]

print(X1)

Y1 = array[:,44]
print(Y1)
estimator = LogisticRegression()
rfe = RFE(estimator, 5)
fit = rfe.fit(X1, Y1)

print("Num Features: %d") % fit.n_features_
print("Selected Features: %s") % fit.support_
print("Feature Ranking: %s") % fit.ranking_

print sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), name))