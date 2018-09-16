import pandas as pd
import numpy as np
import string
import csv
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import sys


puppy_data= pd.read_csv('/Users/milce/Documents/Priya/Softwares/spark-2.0.1-bin-hadoop2.7/playground/b3dataset2.csv')
puppy_data.head(3)


#Get numeric and text columns from Puppy info to join on dogid column

#Get dogid from Trainer info to join on dogid column
# df_train = pd.DataFrame(trainer_data, columns = ['dog_DogID'])
# 
# #Get the columns from puppy-trainer outcome
# df_puppytrainer = pd.DataFrame(puppytrainer_data, columns = ['dog_DogID', 'dog_SubStatusCode','dog_Sex','dbc_DogBreedDescription','dbcc_ColorDescription',])
# 
# 
# #Merge puppyinfo and puppytrainer on dogid column
# merge_puppy_and_puptrainer = pd.merge(puppy_data, df_puppytrainer,  how='inner', left_on='ogr_DogID', right_on='dog_DogID')
# 
# 
# del merge_puppy_and_puptrainer['dog_DogID']
# del merge_puppy_and_puptrainer['ogr_DogID']
# 
# merge_puppy_and_puptrainer.shape

 
puppy_data["ExerciseType"] = puppy_data["ExerciseType"].str.replace('[^\w\s]','')
puppy_data["FoodType"] = puppy_data["FoodType"].str.replace('[^\w\s]','')


#Generate final file
puppy_data.to_csv('/Users/milce/Documents/Priya/Softwares/spark-2.0.1-bin-hadoop2.7/playground/b3dataset2.csv', index= None)
