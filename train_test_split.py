from sklearn.model_selection import train_test_split as t_t_s
from application_logging  import logger

class train_test_split:
    def __init__(self,preprocessed_df):
        self.df=preprocessed_df
    def  split_and_save_file(self):
        X_train, X_test, y_train, y_test=t_t_s(self.df[['content']],self.df[['tag']],stratify=self.df[['tag']],test_size=0.1,random_state=10)
        train_df=X_train
        train_df['tag']=y_train['tag']
        # train_df.to_csv('train_df.csv',index=False)

        test_df=X_test
        test_df['tag']=y_test['tag']
        # test_df.to_csv('test_df.csv',index=False)

        return train_df,test_df

def traintestsplit(preprocessed_df):

    log = logger.App_Logger()
    log.log('loggings.txt','train test split started')
    tts=train_test_split(preprocessed_df)
    train_df,test_df=tts.split_and_save_file()
    log.log('loggings.txt','splitting of data set into the training and test datasets done.')
    return train_df,test_df






