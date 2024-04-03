from flask import Flask, redirect, request, render_template, url_for,jsonify
import re,os
import pandas as pd
import numpy as np
import joblib
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

iris = load_iris()
X = iris.data
y = iris.target

# Preprocessing: Scale features and split dataset
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train KNN classifier
k = 3  # Number of neighbors
knn_classifier = KNeighborsClassifier(n_neighbors=k)
knn_classifier.fit(X_train, y_train)

# Basic user authentication (for demonstration purposes)
users = {
    'admin': '123',
    'ids'  : 'test'
}

# Intrusion Detection System middleware
class IntrusionDetectionMiddleware:
    def __init__(self, app):
        self.app = app

    def __call__(self, environ, start_response):
        self.detect_intrusion(environ)
        return self.app(environ, start_response)

    def detect_intrusion(self, environ):
        # Example: Detecting SQL Injection
        if re.search(r"(select|union|insert|drop|alter|create|update|delete)\s", environ.get('QUERY_STRING', '')):
            self.log_intrusion(environ, "SQL Injection Attempt")

    def log_intrusion(self, environ, message):
        with open("intrusion.log", "a") as log_file:
            log_file.write(f"Intrusion Detected: {message}\n")
            log_file.write(f"Request details: {environ}\n\n")

# Apply the Intrusion Detection Middleware
app.wsgi_app = IntrusionDetectionMiddleware(app.wsgi_app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login')
def login():
    return render_template('login.html')
@app.route('/index')
def home():
    return render_template('index.html')

@app.route('/authenticate', methods=['POST'])
def authenticate():
    username = request.form['username']
    password = request.form['password']
    if username in users and users[username] == password:
        return redirect(url_for('upload'))

    else:
        return redirect(url_for('login'))
    
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Handle file upload logic here
        uploaded_file = request.files['file']
        # Save the file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
        uploaded_file.save(file_path)

        data = pd.read_csv(file_path)

        # Perform data preprocessing
        # Encode categorical variables
        label_encoders = {}
        for column in data.select_dtypes(include=['object']).columns:
           label_encoders[column] = LabelEncoder()
           data[column] = label_encoders[column].fit_transform(data[column])

        # Train a KMeans clustering model
        kmeans = KMeans(n_clusters=2)  # Specify the number of clusters
        clusters = kmeans.fit_predict(data)

        # Save the trained clustering model
        model_path = os.path.join(app.config['UPLOAD_FOLDER'], 'clustering_model.pkl')
        joblib.dump(kmeans, model_path)

        return 'File uploaded and clustering model trained successfully! <br><br> <a href="/">Go back to index page</a>'

    return render_template('upload.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array(data['features']).reshape(1, -1)  # Convert features to 2D array
    scaled_features = scaler.transform(features)
    prediction = knn_classifier.predict(scaled_features)[0]
    predicted_class = iris.target_names[prediction]
    return jsonify({'predicted_class': predicted_class})

if __name__ == '__main__':
    app.run(debug=True)
