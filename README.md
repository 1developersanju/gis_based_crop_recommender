# GeoCrop: Smart Crop Recommendation System

A web-based crop recommendation system that uses machine learning and geospatial intelligence to provide location-specific crop recommendations.

## Features

- **Interactive Map Interface**: Tap on any location to get land parcel details
- **Soil Analysis**: Automatic soil data retrieval from TN Agrisnet API
- **Climate Data**: Integration with climate APIs for temperature, humidity, and rainfall
- **ML-Powered Recommendations**: Uses Random Forest/XGBoost models for crop recommendations
- **Fallback System**: Rule-based recommendations when ML model is not available
- **Modern UI**: Glassmorphic design with responsive layout

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. (Optional) Train the Model

If you have the `Crop_recommendation.csv` file, place it in the project directory. The system will automatically train the model on first run.

Alternatively, you can manually train by:
```python
from crop_recommender import CropRecommender
recommender = CropRecommender(
    model_path='crop_model.pkl',
    csv_path='Crop_recommendation.csv'
)
recommender.train_model()
```

### 3. Run the Flask Server

```bash
python app.py
```

The server will start on `http://127.0.0.1:5000`

### 4. Open the Web Interface

Open `index.html` in your browser, or navigate to `http://127.0.0.1:5000` if serving through Flask.

## API Endpoints

### POST `/recommend`

Get crop recommendations based on soil and climate data.

**Request Body:**
```json
{
  "soil": {
    "ph": 7.2,
    "nitrogen": 150,
    "phosphorus": 25,
    "potassium": 180,
    "moisture": 45
  },
  "climate": {
    "temperature_2m_max": 32,
    "temperature_2m_min": 24,
    "humidity": 65,
    "rainfall": 1200
  }
}
```

**Response:**
```json
{
  "success": true,
  "primary_crop": "Rice",
  "recommendations": [
    {
      "crop": "Rice",
      "suitability": 85.5,
      "confidence": 85.5,
      "season": "Kharif",
      "roi": 45,
      "reason": "Based on ml_model analysis with 85.5% confidence"
    },
    ...
  ],
  "method": "ml_model",
  "input_parameters": {
    "N": 150,
    "P": 25,
    "K": 180,
    "temperature": 28,
    "humidity": 65,
    "ph": 7.2,
    "rainfall": 1200
  }
}
```

### POST `/proxy`

Get land parcel details from TN GIS API.

**Request:**
```
POST /proxy
Content-Type: application/x-www-form-urlencoded

latitude=11.759570&longitude=79.325697&up=
```

### POST `/soil-card`

Get soil card data from TN Agrisnet API.

**Request:**
```
POST /soil-card
Content-Type: application/x-www-form-urlencoded

district=569&block=6111&village=644515&survey_no=15
```

### GET `/climate`

Get climate data for coordinates.

**Request:**
```
GET /climate?lat=11.759570&lng=79.325697
```

## Model Information

The system uses:
- **Random Forest Classifier** (primary model) - 99% accuracy
- **XGBoost** (alternative) - 99% accuracy
- **Rule-based fallback** - When ML model is not available

### Input Features:
- N (Nitrogen) - 0-140 ppm
- P (Phosphorus) - 0-145 ppm
- K (Potassium) - 0-205 ppm
- Temperature - 0-50°C
- Humidity - 0-100%
- pH - 0-14
- Rainfall - 0-300 mm

### Supported Crops:
The model can recommend 22+ crops including:
- Rice, Wheat, Maize
- Cotton, Sugarcane
- Groundnut, Soybean
- Fruits: Mango, Apple, Banana, Orange, etc.
- Pulses: Lentil, Chickpea, Blackgram, etc.

## Project Structure

```
.
├── app.py                 # Flask backend server
├── crop_recommender.py    # ML model and recommendation logic
├── index.html            # Frontend web interface
├── requirements.txt      # Python dependencies
├── crop_model.pkl        # Trained model (generated)
└── README.md            # This file
```

## Usage

1. **Select Location**: Click on the map or use the search box to find your land
2. **View Land Details**: The system automatically fetches land parcel boundaries and metadata
3. **Check Soil & Climate**: Soil and climate data are automatically loaded
4. **Get Recommendations**: Click "Get Crop Recommendations" to see suitable crops

## Development

### Adding New Crops

Edit the `_fallback_predict` method in `crop_recommender.py` to add new crop rules.

### Training New Models

Place your training CSV file in the project directory with columns:
- N, P, K, temperature, humidity, ph, rainfall, label

The system will automatically train on first run.

## License

This project is for educational/research purposes.

## Notes

- The system works with or without the ML model (uses fallback if model not available)
- Soil data is fetched from TN Agrisnet API (Tamil Nadu specific)
- Climate data can be integrated with Open-Meteo or similar APIs
- The frontend uses Google Maps API (requires API key)

# Gis_based_crop_recommender
# gis_based_crop_recommender
# gis_based_crop_recommender
