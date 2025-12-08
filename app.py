from flask import Flask, render_template, request
import joblib
import numpy as np

# Corrected: Use __name__ instead of _name_
app = Flask(__name__)

# Load the model
model = joblib.load('cervical_spondylosis_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user inputs from the form
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    
    # Make prediction
    prediction = model.predict(final_features)
    
    # Map prediction to result
    result = "Negative" if prediction[0] == 1 else "Positive"
    
    # Pass input values back to the frontend
    input_values = {
        'age': request.form['age'],
        'neck_pain': request.form['neck_pain'],
        'stiffness': request.form['stiffness'],
        'numbness': request.form['numbness'],
        'headache': request.form['headache'],
        'sitting_hours': request.form['sitting_hours']
    }
    
    return render_template('index.html', prediction_text=f'Prediction: {result}', **input_values)

# Corrected: Use __name__ and __main__
if __name__ == '__main__':
    app.run(debug=True)