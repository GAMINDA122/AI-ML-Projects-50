from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib
import tensorflow as tf
import json
import os

app = Flask(__name__)

# Load the trained model and scaler
try:
    model = tf.keras.models.load_model("iris_tf_model.keras")
    scaler = joblib.load("scaler.joblib")
    
    # Load label mapping
    if os.path.exists("label_map.json"):
        with open("label_map.json", "r") as f:
            label_names = json.load(f)
    else:
        # Fallback to default iris labels
        label_names = ['setosa', 'versicolor', 'virginica']
    
    print("✅ Model and scaler loaded successfully!")
    print(f"📊 Model expects {model.input_shape[1]} features")
    print(f"🏷️ Labels: {label_names}")
    
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None
    scaler = None
    label_names = ['setosa', 'versicolor', 'virginica']

@app.route('/')
def home():
    try:
        return render_template('index.html')
    except Exception as e:
        return f"""
        <h1>🚨 Error loading template!</h1>
        <p>{e}</p>
        <p>Make sure <code>templates/index.html</code> exists.</p>
        """

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None or scaler is None:
            return jsonify({
                'error': 'Model not loaded properly. Please check your model files.'
            }), 500
        
        # Get data from request
        data = request.get_json()
        
        # Validate input data
        required_fields = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400
            if not isinstance(data[field], (int, float)):
                return jsonify({'error': f'Invalid data type for {field}'}), 400
        
        # Prepare features
        features = np.array([[
            data['sepal_length'],
            data['sepal_width'], 
            data['petal_length'],
            data['petal_width']
        ]])
        
        # Validate feature ranges (basic sanity check)
        if np.any(features < 0) or np.any(features > 20):
            return jsonify({'error': 'Feature values seem unrealistic. Please check your inputs.'}), 400
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        predictions = model.predict(features_scaled, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]) * 100)
        
        # Get all probabilities
        probabilities = {}
        for i, label in enumerate(label_names):
            probabilities[label] = round(float(predictions[0][i] * 100), 1)
        
        # Prepare response
        response = {
            'predicted_species': label_names[predicted_class].title(),
            'confidence': round(confidence, 1),
            'probabilities': probabilities,
            'raw_features': features.tolist()[0],
            'scaled_features': features_scaled.tolist()[0]
        }
        
        print(f"🔮 Prediction made: {response['predicted_species']} ({response['confidence']}%)")
        return jsonify(response)
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return jsonify({
            'error': 'An error occurred during prediction. Please try again.'
        }), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    status = {
        'status': 'healthy',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None,
        'labels': label_names
    }
    return jsonify(status)

if __name__ == '__main__':
    print("🚀 Starting Iris Classifier Gen Z Web App...")
    print("📱 Open your browser and go to: http://localhost:5000")
    print("🌸 Ready to classify some flowers!")
    
    app.run(debug=True, host='0.0.0.0', port=5000)