from flask import * 
import pickle
import numpy as np

app = Flask(__name__)
cv=pickle.load(open('transform.pkl','rb'))
clf = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def homescreen():
    return render_template('frontend.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    message = request.form['message']
    data = [message]
    vect = cv.transform(data).toarray()
    prediction = clf.predict(vect)
    if prediction==1:
        return render_template('frontend.html',pred='''This Compilance is Preventive''', x="")
    else:
        return render_template('frontend.html', pred='This Compilance is Detective', x="")


if __name__ == "__main__":
    app.run(debug=True)