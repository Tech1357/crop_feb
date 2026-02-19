# Crop Recommendation System using Machine Learning 🌾

A web-based application that recommends the most suitable crops to grow based on soil and environmental parameters using machine learning algorithms.

## Features

- **Intelligent Crop Prediction**: Uses machine learning to analyze soil nutrients (N, P, K), temperature, humidity, pH level, and rainfall data
- **Top 3 Recommendations**: Provides the three most suitable crops with confidence percentages
- **Input Validation**: Ensures all input parameters are within realistic agricultural ranges
- **Responsive Web Interface**: Clean, modern UI built with Bootstrap and custom CSS
- **Real-time Processing**: Instant predictions without page reload

## Technologies Used

- **Backend**: Python Flask
- **Machine Learning**: scikit-learn, joblib
- **Data Processing**: pandas, numpy
- **Frontend**: HTML5, CSS3, Bootstrap 5
- **Model Persistence**: Pickle for saving trained models and scalers

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Tech1357/Crop_Recommendation_System_using_ML.git
cd Crop_Recommendation_System_using_ML
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Flask application:
```bash
python app.py
```

2. Open your web browser and navigate to `http://localhost:5000`

3. Enter the following parameters:
   - Nitrogen (kg/ha): 0-140
   - Phosphorus (kg/ha): 5-145
   - Potassium (kg/ha): 5-205
   - Temperature (°C): 8-43
   - Humidity (%): 14-99
   - pH: 3.5-9.5
   - Rainfall (mm): 20-298

4. Click "Predict" to get crop recommendations

## Dataset

The model is trained on the `Crop_recommendation.csv` dataset containing agricultural data for 22 different crops including:
- Rice, Maize, Jute, Cotton, Coconut, Papaya
- Orange, Apple, Muskmelon, Watermelon, Grapes
- Mango, Banana, Pomegranate, Lentil, Blackgram
- Mungbean, Mothbeans, Pigeonpeas, Kidneybeans
- Chickpea, Coffee

## Model Details

- **Algorithm**: Random Forest Classifier (or similar ensemble method)
- **Preprocessing**: MinMax scaling followed by Standard scaling
- **Features**: 7 input parameters
- **Output**: Probability distribution across 22 crop classes

## Project Structure

```
Crop_Recommendation_System_using_ML/
├── app.py                      # Main Flask application
├── Crop_Recommendation_System_using_ML.ipynb  # Jupyter notebook with model training
├── Crop_recommendation.csv     # Training dataset
├── requirements.txt            # Python dependencies
├── debug_prediction.py         # Debugging script
├── test_model.py              # Model testing script
├── model.pkl                  # Trained ML model
├── standard_scaler.pkl        # Standard scaler for features
├── minmax_scaler.pkl          # MinMax scaler for features
├── static/                    # Static files (CSS, JS, images)
├── templates/
│   └── index.html            # Main web interface
└── __pycache__/              # Python cache files
```

## Input Validation Ranges

The application validates inputs to ensure they fall within realistic agricultural ranges:

- **Nitrogen**: 0-140 kg/ha
- **Phosphorus**: 5-145 kg/ha
- **Potassium**: 5-205 kg/ha
- **Temperature**: 8-43°C
- **Humidity**: 14-99%
- **pH**: 3.5-9.5
- **Rainfall**: 20-298 mm

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is open source and available under the [MIT License](LICENSE).

## Acknowledgments

- Dataset source: Agricultural research data
- Built with Flask web framework
- UI design inspired by modern agricultural technology interfaces