from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import pickle
from application_logging  import logger
from file_operations import file_operations


class vectorization_class:
    def __init__(self,train_df,test_df): #we  will provide  the path of train_df and test_df
        self.df_train=train_df
        self.df_test=test_df

    def tf_idf(self):


        tfidf = TfidfVectorizer()
        matrix_train = tfidf.fit_transform(self.df_train['content']).todense()
        pickle.dump(tfidf, open("tf_idf.pkl", "wb"))



        df1=pd.DataFrame(matrix_train,columns=tfidf.get_feature_names())
        df1['tag']=self.df_train.reset_index(drop=True)['tag']

        matrix_test = tfidf.transform(self.df_test['content']).todense()


        df2 = pd.DataFrame(matrix_test, columns=tfidf.get_feature_names())
        df2['tag'] = self.df_test.reset_index(drop=True)['tag']

        del_current_train_files = file_operations('final_train.csv')
        status = del_current_train_files.delete_if_exists()

        logger_object=logger.App_Logger()
        if(status==1):
            logger_object.log('loggings.txt', 'existing training and testing files found and deleted, saving new created files...')
        else:
            logger_object.log('loggings.txt','no existing training and testing files , saving new created files...')

        df1.to_csv('final_train.csv')


        del_current_test_files = file_operations('final_test.csv')
        status = del_current_test_files.delete_if_exists()

        df2.to_csv('final_test.csv')

        return df1,df2



    # def feature_selection(self,df1,df2):
    #     best=SelectKBest(score_func=chi2,k=200)
    #     df_new_train=best.fit_transform(df1.drop('tag',axis=1),df1['tag'])
    #     df_new_train['tag']=df1['tag']
    #     df_new_test=best.transform(df2.drop('tag',axis=1))
    #     df_new_test=df2['tag']
    #     df_new_train.to_csv('final_train.csv')
    #     df_new_test.to_csv('final_test.csv')
    #     # chi_scores=chi2(df1.drop('tag',axis=1),df1['tag'])
    #     # p_values = pd.Series(chi_scores[1], index=df1.drop('tag',axis=1).columns)
    #     # p_values.sort_values(ascending=False, inplace=True)
    #     return df_new_train,df_new_test




    # def pca(self,df1,df2):
    #     # print(df1.shape())
    #     le=LabelEncoder()
    #
    #     log=logger.App_Logger()
    #     log.log('loggings.txt', 'starting principal component analysis')
    #
    #     model=PCA(n_components=500)
    #     lst_train=model.fit_transform(df1.drop(['tag'],axis=1))
    #     pickle.dump(model, open("pca.pkl", "wb"))
    #
    #     log.log('loggings.txt', 'Pca model created and exported to pca.pkl now exporting data to final_train.csv and final_test.csv')
    #
    #     new_df_train=pd.DataFrame(lst_train)
    #     new_df_train['tag']=df1['tag']
    #     new_df_train['tag']=le.fit_transform(df1['tag'])
    #     new_df_train.to_csv('final_train.csv')
    #
    #     lst_test=model.transform(df2.drop(['tag'],axis=1))
    #     new_df_test = pd.DataFrame(lst_test)
    #     new_df_test['tag'] = df2['tag']
    #     new_df_test['tag'] = le.transform(df2['tag'])
    #     new_df_test.to_csv('final_test.csv')
    #
    #
    #     log.log('loggings.txt', 'exported successfully')
    #     return new_df_train,new_df_test


def vectorize(train_df,test_df):
    log = logger.App_Logger()

    log.log('loggings.txt', 'vectorization started')

    del_current_train_files = file_operations('tf_idf.pkl')
    status = del_current_train_files.delete_if_exists()

    if (status == 1):
        log.log('loggings.txt','existing tfidf model found and deleted, saving new created model...')
    else:
        log.log('loggings.txt', 'no existing tfidf model found , saving new model...')

    log.log('loggings.txt', 'vectorization(tfidf) of train and test files done and tfidf model saved at tfidf.pkl')


    vc=vectorization_class(train_df,test_df)

    df_final_train,df_final_test=vc.tf_idf()





    return df_final_train,df_final_test



