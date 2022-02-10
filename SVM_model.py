import pickle
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score,confusion_matrix,classification_report
from application_logging  import logger
from file_operations import file_operations

class model_training:
    def __init__(self,df_train,df_test):
        self.train_df=df_train
        self.test_df=df_test

    def svm_training(self):
        log=logger.App_Logger()
        log.log('loggings.txt', 'model creation started')
        svc=SVC( kernel= 'linear')
        svc.fit(self.train_df.drop(['tag'],axis=1),self.train_df['tag'])
        predicted_y=svc.predict(self.test_df.drop(['tag'],axis=1))
        print(roc_auc_score(self.test_df['tag'],predicted_y))
        print(confusion_matrix(self.test_df['tag'],predicted_y))
        print(classification_report(self.test_df['tag'],predicted_y))

        del_current_model=file_operations('model.pkl')
        status=del_current_model.delete_if_exists()
        if(status==1):
            log.log('loggings.txt', 'existing model found and deleted, saving new model...')
        else:
            log.log('loggings.txt','no existing model found, saving new model...')

        pickle.dump(svc, open("model.pkl", "wb"))
        log.log('loggings.txt', 'model creation done and model saved in model.pkl file')


# nb=MultinomialNB()
# nb.fit(df_train.drop(['Unnamed: 0','tag'],axis=1),df_train['tag'])
# predicted_y_nb=nb.predict(df_test.drop(['Unnamed: 0','tag'],axis=1))
# print(roc_auc_score(df_test['tag'],predicted_y_nb))
# print(confusion_matrix(df_test['tag'],predicted_y_nb))
# print(classification_report(df_test['tag'],predicted_y_nb))