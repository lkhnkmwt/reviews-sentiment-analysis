from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk.corpus import stopwords
import re
from application_logging import logger
from sklearn.preprocessing import LabelEncoder
from nltk import word_tokenize

class data_preprocessing:
    def __init__(self,aggregated_df):
        self.df=aggregated_df
        self.App_logger=logger.App_Logger()

    def regex_filtering(self,text):
        '''this function takes the text as input and replaces the non alphabets with whitespace and returns it'''
        return re.sub(r'[^a-zA-Z]',' ',text)

    def Word_tokenize(self,text):
        lst= word_tokenize(text)
        return (' '.join(lst))



    # def email_filter(self,text):
    #     '''this fuction seperate the main content of email from the other things like receiver's address
    #         sender's address , subject , date, etc.'''
    #     words=['return','path','delivered','to','received','localhost','by','for','']


    def lowerandsplit(self,text):
        '''this function takes text as input and return after lowering and splitting into list'''
        return re.split(r'\s+',text.lower())

    # def lemm(self,lst):
    #     ''' this function takes paragraph words' list as input and lematize it followed by
    #     removing stopwords and finally joins the words of list to return the joined paragraph'''
    #     lematizzer=WordNetLemmatizer()
    #     new_lst=[lematizzer.lemmatize(word) for word in set(lst) if word not in stopwords.words('english')]
    #     return ' '.join(new_lst)

    def stem(self,lst):
        ''' this function takes paragraph words' list as input and stem it followed by
        removing stopwords and finally joins the words of list to return the joined paragraph'''
        porter = PorterStemmer()
        new_lst=[porter.stem(word) for word in lst if word not in stopwords.words('english')]
        return ' '.join(new_lst)




    def Labelencode(self,df2):
        '''this function performs the label encoding of tag column'''
        le=LabelEncoder()
        df2['tag']=le.fit_transform(df2['tag'])




def text_preprocess(aggregated_df):
    new = []
    object1=data_preprocessing(aggregated_df)
    df=object1.df
    object1.App_logger.log('loggings.txt', 'data preprocessing started')
    for para in df['content']:
        filtered=object1.regex_filtering(para)
        tokenized=object1.Word_tokenize(filtered)
        lowered=object1.lowerandsplit(tokenized)
        stemmed=object1.stem(lowered)
        new.append(stemmed)


    df['content']=new

    object1.Labelencode(df)

    object1.App_logger.log('loggings.txt','data preprocessing was successful')

    return df




    # def tf_idf(self):
    #     '''using tf_idf for vectoring purpose'''


