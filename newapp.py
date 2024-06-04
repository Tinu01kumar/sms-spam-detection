
from flask import Flask, render_template, request, redirect, url_for
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from pymongo import MongoClient

nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

# Load vectorizer and models
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
models = {
    # "LR_model.pkl": pickle.load(open('LR_model.pkl', 'rb')),
    # "KN_model.pkl": pickle.load(open('KN_model.pkl', 'rb')),
    # "SVC_model.pkl": pickle.load(open('SVC_model.pkl', 'rb')),
    # "mnb_clf_model.pkl": pickle.load(open('mnb_clf_model.pkl', 'rb')),
    # "gnb_clf_model.pkl": pickle.load(open('gnb_clf_model.pkl', 'rb')),
    # "bnb_clf_model.pkl": pickle.load(open('bnb_clf_model.pkl', 'rb')),
    "DT_model.pkl": pickle.load(open('DT_model.pkl', 'rb')),
    # "RF_model.pkl": pickle.load(open('RF_model.pkl', 'rb')),
    # "AdaBoost_model.pkl": pickle.load(open('AdaBoost_model.pkl', 'rb')),
    # "BgC_model.pkl": pickle.load(open('BgC_model.pkl', 'rb')),
    # "ETC_model.pkl": pickle.load(open('ETC_model.pkl', 'rb')),
    # "stacking_clf_model.pkl" : pickle.load(open("stacking_clf_model.pkl", 'rb')),
    # "voting_clf_model.pkl" : pickle.load(open("voting_clf_model.pkl" , 'rb'))



}

app = Flask(__name__)

# MongoDB connection
client = MongoClient("mongodb+srv://kashyap01edu:Mri2oJrILvBslxxK@cluster0.dtkbmae.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client["SPAM_MESSAGES"]
collection = db["data"]

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = [i for i in text if i.isalnum()]
    y = [ps.stem(i) for i in y if i not in stopwords.words('english') and i not in string.punctuation]
    return " ".join(y)

@app.route('/')
def index():
    return render_template('index.html', ans="", content="", spam_probability="")

@app.route('/predict', methods=['POST'])
def predict_sms():
    global prediction_result

    content = request.form.get('content')
    selected_model = request.form.get('model')

    transformed_sms = transform_text(content)
    vector_input = tfidf.transform([transformed_sms])

    # Convert sparse matrix to dense array if the selected model is SVC
    if selected_model in ['gnb_clf_model.pkl', 'stacking_clf_model.pkl', 'voting_clf_model.pkl', 'SVC_model.pkl']:
        vector_input = vector_input.toarray()

    # Use the selected model for prediction
    model = models[selected_model]
    result = model.predict(vector_input)[0]
    probabilities = model.predict_proba(vector_input)
    spam_probability = round(probabilities[0, 1] * 100)

    if result == 1:
        ans = "spam"
    else:
        ans = "ham"

    prediction_result = ans

    return render_template('index.html', ans=ans, content=content, spam_probability=spam_probability)

@app.route('/feedback', methods=['POST'])
def user_feedback():
    global prediction_result

    content = request.form.get('content')
    user_feedback = request.form.get('user_feedback')

    ans = request.form.get('prediction_result')
    if user_feedback == 'incorrect':
        target = 'spam' if ans == 'ham' else 'ham'
        message = {'text': content, 'target': target}
        collection.insert_one(message)

    return redirect(url_for('index'))

if __name__ == '__main__':
    #app.run(host="0.0.0.0", port=8080)
    app.run(debug=True)





