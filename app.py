from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load model and encoder
dt_model = joblib.load('dt_classifier.pkl')
encoder = joblib.load('encoder.pkl')  # Memuat encoder yang telah disimpan

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get input from form
        user_id = request.form['user_id']
        product_id = request.form['product_id']
        
        # Prepare input data
        input_data = pd.DataFrame({'USER_ID': [user_id], 'PRODUCT_ID': [product_id]})
        
        # Encode input features
        input_encoded = encoder.transform(input_data)
        
        # Predict rating
        rating_prediction = dt_model.predict(input_encoded)
        
        return render_template('result.html', user_id=user_id, product_id=product_id, rating_prediction=rating_prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
