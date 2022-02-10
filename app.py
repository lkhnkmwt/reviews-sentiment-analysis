from flask import Flask,render_template,request
from flask_cors import CORS,cross_origin
from data_preprocessing import data_preprocessing
import pickle
import pandas as pd

app=Flask(__name__)
CORS(app)

@app.route('/',methods=['GET'])
@cross_origin()
def homepage():
    return render_template('index.html')



@app.route('/prediction',methods=['GET','POST'])
def predict():
    if(request.method=='POST'):
        review = request.form['review']
        text_preprocess_object=data_preprocessing(review)
        filtered=text_preprocess_object.regex_filtering(review)
        tokenized=text_preprocess_object.Word_tokenize(filtered)
        lowered=text_preprocess_object.lowerandsplit(tokenized)
        stemmed=text_preprocess_object.stem(lowered)
        tfidf_model=pickle.load(open("tf_idf.pkl", "rb"))
        X=tfidf_model.transform(pd.DataFrame([stemmed],columns=['content'])['content']).todense()
        X_df = pd.DataFrame(X, columns=tfidf_model.get_feature_names())
        svm_model=pickle.load(open("model.pkl", "rb"))
        prediction=svm_model.predict(X_df)
        if(prediction==0):
            return 'Negative'
        elif(prediction==1):
            return 'Positive'
        else:
            return 'error'


if __name__ == '__main__':
    app.run()