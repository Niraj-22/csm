from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['STATIC_FOLDER'] = 'static'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "No file part"

    file = request.files['file']

    if file.filename == '':
        return "No selected file"

    if file:
        try:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Perform actions without python-script
            result, rfm_plot_path = process_data(file_path)

            return render_template('dashboard.html', result=result, rfm_plot=rfm_plot_path)

        except Exception as e:
            # Log the exception or return a user-friendly error message
            return f"An error occurred: {str(e)}"

def process_data(file_path):
    # Read the CSV file
    df_data = pd.read_csv(file_path)

    # Check for NaN values
    if df_data.isnull().sum().any():
        # Drop rows with NaN values
        df_data.dropna(subset=['Quantity', 'Price'], inplace=True)
        
        if df_data.empty:
            return "Dataset is empty after dropping NaN values. Please check your data.", None

        # Example: Calculate Recency, Frequency, Monetary
        today_date = pd.to_datetime('2011-12-11')
        df_data['InvoiceDate'] = pd.to_datetime(df_data['InvoiceDate'])
        df_data['TotalPrice'] = df_data['Quantity'] * df_data['Price']
        df_data['Recency'] = (today_date - df_data['InvoiceDate']).dt.days
        rfm = df_data.groupby('Customer ID').agg({'Recency': 'min', 'Invoice': 'nunique', 'TotalPrice': 'sum'})

        # Handle NaN values
        imputer = SimpleImputer(strategy='mean')
        rfm[['Recency', 'TotalPrice']] = imputer.fit_transform(rfm[['Recency', 'TotalPrice']].values)

        # Example: Apply K-Means clustering
        rfm_log = np.log1p(rfm[['Recency', 'TotalPrice']])
        scaler = StandardScaler()
        rfm_scaled = scaler.fit_transform(rfm_log)

        kmean_model = KMeans(n_clusters=4, init='k-means++', max_iter=1000, random_state=20)
        kmean_model.fit(rfm_scaled)
        rfm['Cluster'] = kmean_model.labels_

        # Generate and save RFM plot
        rfm_plot_path = os.path.join(app.config['STATIC_FOLDER'], 'rfm_plot.png')
        generate_rfm_plot(rfm, rfm_plot_path)

        return rfm.describe().to_html(), rfm_plot_path

def generate_rfm_plot(rfm, plot_path):
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=rfm, x='Recency', y='TotalPrice', hue='Cluster', palette='viridis', s=100)
    plt.title('RFM Clustering')
    plt.savefig(plot_path)

if __name__ == '__main__':
    app.run(debug=True)
