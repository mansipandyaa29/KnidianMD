from flask import Flask,request,render_template

# from sklearn.preprocessing import StandardScaler

from src.predict.predict_pipeline import PredictPipeline

application = Flask(__name__)
app = application

##Route for a home page

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = request.form.get('medical_history')
        print(data)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(data)

        return render_template('home.html',results = results)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
