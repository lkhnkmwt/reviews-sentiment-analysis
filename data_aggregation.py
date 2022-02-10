import os
import pandas as pd
from application_logging import logger

class dataagg:
    def __init__(self,path):
        self.path=path
        self.logging_path='loggings.txt'


def extract_and_combine():
    '''this function extracts the spam and ham data from training
    data folder and combine them.Finally save it to aggrigated_data folder'''

    try:
        object=dataagg('training data/')

        df = pd.DataFrame(columns=['content', 'tag'])

        for hamspam in os.listdir(object.path):
            for i in (os.listdir(object.path + hamspam)):
                with open(object.path + hamspam + '/' + i, 'r', errors='ignore', newline='\n') as f:
                    df1 = pd.DataFrame([[f.read(), hamspam]], columns=['content', 'tag'])
                    df = df.append(df1, ignore_index=True)

        if os.path.exists('aggrigated_data\\aggregated.csv'): #remove aggregated.csv if already exists
            # print('is exists')
            os.remove('aggrigated_data\\aggregated.csv')
            logger.App_Logger().log(object.logging_path,"existed aggregated data found and deleted")
            df.to_csv('aggrigated_data\\aggregated.csv',index=False)
            return df


        else:
            df.to_csv(os.path.join('aggrigated_data\\aggregated.csv'),index=False)
            return df
            logger.App_Logger().log(object.logging_path, "no aggregated data file found, created new file")

        logger.App_Logger().log(object.logging_path,"successfully aggregated the data from training data folder to aggrigated_data folder")

    except Exception as e:
        logger.App_Logger().log(object.logging_path,"aggregating the data from training data folder to aggrigated_data folder was unsuccessful"+e)
        print(e)








