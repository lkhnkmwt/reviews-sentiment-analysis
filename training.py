import data_preperation
from train_test_split import traintestsplit
from vectorization import vectorize
import pandas as pd
from SVM_model  import model_training



aggregated_df=pd.read_csv('aggrigated_data/reviews_data.csv') # reading aggregated file

aggregated_df.dropna(inplace=True)

preperation=data_preperation.data_prep()

preprocessed_df=preperation.data_preprocess(aggregated_df)

train_df,test_df=traintestsplit(preprocessed_df)

final_df_train,final_df_test=vectorize(train_df,test_df)

train=model_training(final_df_train,final_df_test)

train.svm_training()
