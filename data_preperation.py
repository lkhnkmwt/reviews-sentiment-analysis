from data_preprocessing import  text_preprocess
class data_prep:
    def __init__(self):
        pass

    def data_preprocess(self,aggregated_df):
        df1=text_preprocess(aggregated_df)
        # df1.to_csv('final_csv',index=False)
        return df1

    # def email_preprocess(self):
    #     check_words=[]

# saving_final_file=data_prep()
# saving_final_file.data_preprocess()
