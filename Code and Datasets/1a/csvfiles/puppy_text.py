
puppy_data= pd.read_csv('/Users/milce/Documents/SJSU/CS274-XML&WI/Milestone3/PuppyTextOnly_new.csv')
trainer_data= pd.read_csv('/Users/milce/Documents/SJSU/CS274-XML&WI/Milestone3/TrainerInfo1.csv')
puppytrainer_data = pd.read_csv('/Users/milce/Documents/SJSU/CS274-XML&WI/Milestone3/PuppyTrainerOutcome1.csv')
puppy_data.head(3)
trainer_data.head(3)
puppytrainer_data.head(3)

#Get dogid from Trainer info to join on dogid column
df_train = pd.DataFrame(trainer_data, columns = ['dog_DogID'])

#Get the columns from puppy-trainer outcome
df_puppytrainer = pd.DataFrame(puppytrainer_data, columns = ['dog_DogID','dog_Sex','dog_SubStatusCode','dbc_DogBreedDescription','dbcc_ColorDescription'])

#Merge puppyinfo and puppytrainer on dogid column
merge_puppy_and_puptrainer = pd.merge(puppy_data, df_puppytrainer,  how='inner', left_on='ogr_DogID', right_on='dog_DogID')


del merge_puppy_and_puptrainer['dog_DogID']
del merge_puppy_and_puptrainer['ogr_DogID']

merge_puppy_and_puptrainer.shape

#Cleaning on Statuscode columns
merge_puppy_and_puptrainer['dog_SubStatusCode']= merge_puppy_and_puptrainer.loc[:,'dog_SubStatusCode'].replace([1,2,3,17,21,27,28,29,37,45,50,66,76,97,111,154,159,160,161,162,166,167,171,172,173,174,175,179,181,184,187,188],
                        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
merge_puppy_and_puptrainer['dog_SubStatusCode']=merge_puppy_and_puptrainer.loc[:,'dog_SubStatusCode'].replace([23,25,26,27,55,98,99,121,169],[1,1,1,1,1,1,1,1,1])

merge_puppy_and_puptrainer.to_csv('/Users/milce/Documents/SJSU/CS274-XML&WI/Milestone3/Puppy_Text_With_Header.csv', index= None)




#Merge puppyinfo and puppytrainer on dogid column