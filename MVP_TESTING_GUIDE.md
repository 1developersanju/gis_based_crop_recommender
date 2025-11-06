# MVP Testing Guide - Crop Recommendation Endpoint

## Quick Start

### 1. Start the Flask Server

```bash
python app.py
```

The server should start on `http://127.0.0.1:5000`

### 2. Test the Endpoint

#### Option A: Using the Test Script (Recommended)

```bash
python test_recommendation.py
```

This will automatically test the endpoint with sample data.

#### Option B: Using curl

```bash
curl -X POST http://127.0.0.1:5000/recommend \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

#### Option C: Using the Web Interface

1. Open `index.html` in your browser
2. Click on the map to select a location
3. Wait for soil/climate data to load
4. Click "Get Crop Recommendations" button
5. View the recommendations in the panel

## Expected Response

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
      "reason": "Based on rule_based analysis with 85.5% confidence"
    },
    ...
  ],
  "method": "rule_based",
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

## Test Cases

### Test Case 1: Basic Recommendation
```json
{
  "soil": {
    "ph": 7.0,
    "nitrogen": 100,
    "phosphorus": 30,
    "potassium": 50
  },
  "climate": {
    "temperature_2m_max": 30,
    "temperature_2m_min": 22,
    "humidity": 70,
    "rainfall": 150
  }
}
```

### Test Case 2: Rice-Favorable Conditions
```json
{
  "soil": {
    "ph": 6.5,
    "nitrogen": 120,
    "phosphorus": 20,
    "potassium": 40
  },
  "climate": {
    "temperature_2m_max": 32,
    "temperature_2m_min": 24,
    "humidity": 80,
    "rainfall": 200
  }
}
```

### Test Case 3: Cotton-Favorable Conditions
```json
{
  "soil": {
    "ph": 7.5,
    "nitrogen": 80,
    "phosphorus": 40,
    "potassium": 50
  },
  "climate": {
    "temperature_2m_max": 35,
    "temperature_2m_min": 25,
    "humidity": 60,
    "rainfall": 80
  }
}
```

## Troubleshooting

### Issue: "Could not connect to Flask server"
**Solution**: Make sure the server is running:
```bash
python app.py
```

### Issue: "ModuleNotFoundError"
**Solution**: Install dependencies:
```bash
pip install -r requirements.txt
```

### Issue: "Error generating recommendations"
**Solution**: Check the server logs for detailed error messages. The system uses a fallback rule-based system if the ML model is not available, so it should still work.

### Issue: Recommendations seem incorrect
**Solution**: 
- Verify input parameters are in correct ranges:
  - N: 0-140 ppm
  - P: 0-145 ppm
  - K: 0-205 ppm
  - pH: 0-14
  - Temperature: 0-50°C
  - Humidity: 0-100%
  - Rainfall: 0-300 mm

## Current Implementation Status

✅ **Implemented:**
- `/recommend` endpoint in Flask
- Integration with `crop_recommender.py`
- Rule-based fallback system (works without ML model)
- Frontend integration
- Error handling
- CORS support

⚠️ **Optional (for better accuracy):**
- ML model training (requires `Crop_recommendation.csv`)
- Model persistence (`crop_model.pkl`)

## Notes

- The system **works immediately** with the rule-based fallback
- ML model is optional - system will use it if available
- All endpoints support CORS for frontend integration
- The endpoint accepts flexible parameter names (nitrogen/N, phosphorus/P, etc.)

