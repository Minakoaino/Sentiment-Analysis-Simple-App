
from flask import Flask,render_template,url_for,request
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC


app = Flask(__name__)

@app.route('/')
def home():
	    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
        df = pd.read_csv("sentiment_analysis_data.csv")
        	    # Features and Labels
        df['label'] = df['label'].map({'negative': -1, 'neutral': 0, 'positive':1})
        	    # Extract Feature With CountVectorizer

        tfidf = TfidfVectorizer(max_features=45000)
        X = df['Tweet_without_stopwords']
        y = df['label']
        X = tfidf.fit_transform(df['Tweet_without_stopwords'].values.astype('U'))  

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20,  shuffle = True, random_state = 10)
        ovsrc = OneVsRestClassifier(LinearSVC(random_state=0)).fit(X, y)


        if request.method == 'POST':
	        message = request.form['message']
	        data = [message]
	        vect = tfidf.transform(data).toarray()
	        my_prediction = ovsrc.predict(vect)
        return render_template('result.html',prediction = my_prediction)



if __name__ == '__main__':
	app.run(debug=True)