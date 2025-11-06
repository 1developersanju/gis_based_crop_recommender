"""
Crop Recommendation Model
Based on the Jupyter notebook implementation
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Try to import XGBoost, fallback to RandomForest if not available
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠ XGBoost not available, will use RandomForest")

class CropRecommender:
    def __init__(self, model_path='crop_model.pkl', csv_path=None):
        """
        Initialize the crop recommender
        
        Args:
            model_path: Path to saved model pickle file
            csv_path: Path to training CSV file (optional, for training)
        """
        self.model_path = model_path
        self.csv_path = csv_path
        self.model = None
        self.label_encoder = LabelEncoder()
        self.feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        self.is_trained = False
        
        # Try to load existing model
        if os.path.exists(model_path):
            self.load_model()
        elif csv_path and os.path.exists(csv_path):
            # Train model if CSV is available
            self.train_model()
        else:
            # Use fallback model
            self._create_fallback_model()
    
    def load_model(self):
        """Load a pre-trained model from pickle file"""
        try:
            # Check if this is an XGBoost model and if xgboost is available
            if 'XGBoost' in self.model_path or 'xgboost' in self.model_path.lower():
                if not XGBOOST_AVAILABLE:
                    print(f"⚠ XGBoost model file found ({self.model_path}) but xgboost is not installed.")
                    print("   Install with: pip install xgboost")
                    print("   Falling back to rule-based system...")
                    self._create_fallback_model()
                    return
            
            with open(self.model_path, 'rb') as f:
                data = pickle.load(f)
                
                if isinstance(data, dict):
                    # New format: dict with model and label_encoder
                    self.model = data.get('model')
                    self.label_encoder = data.get('label_encoder')
                    if self.label_encoder is None:
                        # Try to recreate label encoder from CSV if available
                        self._recreate_label_encoder()
                else:
                    # Old format: just the model (like XGBoost.pkl from notebook)
                    self.model = data
                    # Try to recreate label encoder from CSV
                    self._recreate_label_encoder()
                
                if self.model is None:
                    raise ValueError("Model is None after loading")
                
                self.is_trained = True
                
                # Determine model type
                if XGBOOST_AVAILABLE:
                    try:
                        model_type = 'XGBoost' if isinstance(self.model, xgb.XGBClassifier) else 'RandomForest'
                    except:
                        model_type = 'ML Model'
                else:
                    model_type = 'RandomForest'
                
                print(f"✓ {model_type} model loaded from {self.model_path}")
                if self.label_encoder and hasattr(self.label_encoder, 'classes_'):
                    print(f"✓ Label encoder loaded with {len(self.label_encoder.classes_)} classes")
                else:
                    print("⚠ Label encoder not available - will need CSV to recreate")
                    
        except ModuleNotFoundError as e:
            if 'xgboost' in str(e).lower():
                print(f"❌ Error: XGBoost model requires xgboost package")
                print("   Install with: pip install xgboost")
                print("   Falling back to rule-based system...")
            else:
                print(f"Error loading model: {e}")
            self._create_fallback_model()
        except Exception as e:
            print(f"Error loading model: {e}")
            import traceback
            traceback.print_exc()
            self._create_fallback_model()
    
    def _recreate_label_encoder(self):
        """Recreate label encoder from CSV file if available, or use known crop list"""
        if self.label_encoder and hasattr(self.label_encoder, 'classes_') and len(self.label_encoder.classes_) > 0:
            return  # Already has classes
        
        # Try to find CSV file
        csv_paths = [self.csv_path] if self.csv_path else []
        csv_paths.extend(['Crop_recommendation.csv', 'crop_recommendation.csv', '/content/Crop_recommendation.csv'])
        
        for csv_path in csv_paths:
            if csv_path and os.path.exists(csv_path):
                try:
                    print(f"Recreating label encoder from {csv_path}...")
                    df = pd.read_csv(csv_path)
                    target = df['label']
                    self.label_encoder = LabelEncoder()
                    self.label_encoder.fit(target)
                    print(f"✓ Label encoder recreated with {len(self.label_encoder.classes_)} classes")
                    return
                except Exception as e:
                    print(f"Error recreating label encoder from {csv_path}: {e}")
                    continue
        
        # Fallback: Use known crop list from notebook (22 crops)
        # This matches the crops in the notebook output
        known_crops = [
            'apple', 'banana', 'blackgram', 'chickpea', 'coconut', 'coffee',
            'cotton', 'grapes', 'jute', 'kidneybeans', 'lentil', 'maize',
            'mango', 'mothbeans', 'mungbean', 'muskmelon', 'orange', 'papaya',
            'pigeonpeas', 'pomegranate', 'rice', 'watermelon'
        ]
        
        try:
            print("Using known crop list for label encoder...")
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(known_crops)
            print(f"✓ Label encoder created with {len(self.label_encoder.classes_)} known crops")
        except Exception as e:
            print(f"⚠ Error creating label encoder with known crops: {e}")
            self.label_encoder = LabelEncoder()
    
    def train_model(self):
        """Train the model using the CSV file - matches notebook implementation"""
        try:
            print(f"Training model from {self.csv_path}...")
            df = pd.read_csv(self.csv_path)
            
            # Extract features and target (exactly as in notebook)
            features = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
            target = df['label']
            
            # Split data (same as notebook: test_size=0.2, random_state=2)
            Xtrain, Xtest, Ytrain, Ytest = train_test_split(
                features, target, test_size=0.2, random_state=2
            )
            
            # Encode labels (exactly as in notebook)
            Ytrain_encoded = self.label_encoder.fit_transform(Ytrain)
            
            # Use XGBoost if available (as in notebook), otherwise RandomForest
            if XGBOOST_AVAILABLE:
                print("Training XGBoost model (as in notebook)...")
                self.model = xgb.XGBClassifier()
                self.model.fit(Xtrain, Ytrain_encoded)
                print("✓ XGBoost model trained successfully")
            else:
                print("Training RandomForest model (XGBoost not available)...")
                self.model = RandomForestClassifier(n_estimators=20, random_state=0)
                self.model.fit(Xtrain, Ytrain_encoded)
                print("✓ RandomForest model trained successfully")
            
            # Save model
            self.save_model()
            self.is_trained = True
            print("✓ Model saved successfully")
            
        except Exception as e:
            print(f"Error training model: {e}")
            import traceback
            traceback.print_exc()
            self._create_fallback_model()
    
    def save_model(self):
        """Save the trained model to pickle file"""
        try:
            with open(self.model_path, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'label_encoder': self.label_encoder
                }, f)
            print(f"✓ Model saved to {self.model_path}")
        except Exception as e:
            print(f"Error saving model: {e}")
    
    def _create_fallback_model(self):
        """Create a simple fallback model based on rule-based logic"""
        print("⚠ Using fallback rule-based recommendation system")
        self.is_trained = False
        self.model = None
    
    def predict(self, N, P, K, temperature, humidity, ph, rainfall):
        """
        Predict crop recommendation
        
        Args:
            N: Nitrogen (0-140)
            P: Phosphorus (0-145)
            K: Potassium (0-205)
            temperature: Temperature in Celsius (0-50)
            humidity: Humidity percentage (0-100)
            ph: pH value (0-14)
            rainfall: Rainfall in mm (0-300)
        
        Returns:
            dict with crop recommendations and details
        """
        # Prepare input data
        input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        
        if self.is_trained and self.model:
            # Use trained model (exactly as in notebook)
            try:
                # Check if label encoder has classes
                if not hasattr(self.label_encoder, 'classes_') or len(self.label_encoder.classes_) == 0:
                    print("⚠ Label encoder not properly initialized, trying to recreate...")
                    self._recreate_label_encoder()
                    if not hasattr(self.label_encoder, 'classes_') or len(self.label_encoder.classes_) == 0:
                        raise ValueError("Label encoder not available - cannot decode predictions")
                
                # Predict (same as notebook: XB.predict(data))
                prediction_encoded = self.model.predict(input_data)[0]
                
                # Inverse transform to get crop name (same as notebook: le.inverse_transform)
                crop = self.label_encoder.inverse_transform([prediction_encoded])[0]
                
                # Get prediction probabilities for all crops
                probabilities = self.model.predict_proba(input_data)[0]
                crop_names = self.label_encoder.classes_
                
                # Get top 5 recommendations sorted by probability
                top_indices = np.argsort(probabilities)[::-1][:5]
                recommendations = []
                
                for idx in top_indices:
                    crop_name = crop_names[idx]
                    prob = probabilities[idx]
                    
                    # Get ROI and season from fallback data if available
                    fallback_data = self._get_crop_metadata(crop_name)
                    
                    recommendations.append({
                        'crop': crop_name,
                        'suitability': round(prob * 100, 2),
                        'confidence': round(prob * 100, 2),
                        'season': fallback_data.get('season', 'Year-round'),
                        'roi': fallback_data.get('roi', 35)
                    })
                
                # Determine method name
                if XGBOOST_AVAILABLE:
                    try:
                        method = 'xgboost' if isinstance(self.model, xgb.XGBClassifier) else 'ml_model'
                    except:
                        method = 'ml_model'
                else:
                    method = 'ml_model'
                
                print(f"✓ {method.upper()} Model prediction: {crop} (confidence: {probabilities[top_indices[0]]*100:.1f}%)")
                
                return {
                    'success': True,
                    'primary_crop': crop,
                    'recommendations': recommendations,
                    'method': method
                }
            except Exception as e:
                print(f"Error in ML prediction: {e}")
                import traceback
                traceback.print_exc()
                print("Falling back to rule-based recommendations...")
                return self._fallback_predict(N, P, K, temperature, humidity, ph, rainfall)
        else:
            # Use fallback
            return self._fallback_predict(N, P, K, temperature, humidity, ph, rainfall)
    
    def _get_crop_metadata(self, crop_name):
        """Get metadata (season, ROI) for a crop"""
        crop_metadata = {
            'Rice': {'season': 'Kharif', 'roi': 45},
            'Wheat': {'season': 'Rabi', 'roi': 40},
            'Cotton': {'season': 'Kharif', 'roi': 38},
            'Sugarcane': {'season': 'Year-round', 'roi': 42},
            'Maize': {'season': 'Kharif', 'roi': 35},
            'Groundnut': {'season': 'Kharif', 'roi': 32},
            'Soybean': {'season': 'Kharif', 'roi': 30},
            'apple': {'season': 'Year-round', 'roi': 50},
            'banana': {'season': 'Year-round', 'roi': 48},
            'blackgram': {'season': 'Kharif', 'roi': 28},
            'chickpea': {'season': 'Rabi', 'roi': 30},
            'coconut': {'season': 'Year-round', 'roi': 40},
            'coffee': {'season': 'Year-round', 'roi': 45},
            'grapes': {'season': 'Year-round', 'roi': 50},
            'jute': {'season': 'Kharif', 'roi': 25},
            'kidneybeans': {'season': 'Kharif', 'roi': 28},
            'lentil': {'season': 'Rabi', 'roi': 30},
            'mango': {'season': 'Year-round', 'roi': 55},
            'mothbeans': {'season': 'Kharif', 'roi': 28},
            'mungbean': {'season': 'Kharif', 'roi': 30},
            'muskmelon': {'season': 'Kharif', 'roi': 35},
            'orange': {'season': 'Year-round', 'roi': 45},
            'papaya': {'season': 'Year-round', 'roi': 40},
            'pigeonpeas': {'season': 'Kharif', 'roi': 32},
            'pomegranate': {'season': 'Year-round', 'roi': 50},
            'watermelon': {'season': 'Kharif', 'roi': 38}
        }
        return crop_metadata.get(crop_name, {'season': 'Year-round', 'roi': 35})
    
    def _fallback_predict(self, N, P, K, temperature, humidity, ph, rainfall):
        """Fallback rule-based recommendation system"""
        recommendations = []
        
        # Rule-based crop recommendations based on conditions
        crop_rules = [
            {
                'crop': 'Rice',
                'suitability': self._calculate_suitability(N, P, K, temperature, humidity, ph, rainfall, 
                    ideal_N=(50, 120), ideal_P=(10, 50), ideal_K=(20, 50),
                    ideal_temp=(20, 35), ideal_humidity=(70, 90), ideal_ph=(5.5, 7.5), ideal_rainfall=(150, 300)),
                'season': 'Kharif',
                'roi': 45
            },
            {
                'crop': 'Wheat',
                'suitability': self._calculate_suitability(N, P, K, temperature, humidity, ph, rainfall,
                    ideal_N=(50, 100), ideal_P=(20, 40), ideal_K=(20, 40),
                    ideal_temp=(15, 25), ideal_humidity=(40, 60), ideal_ph=(6.0, 7.5), ideal_rainfall=(30, 100)),
                'season': 'Rabi',
                'roi': 40
            },
            {
                'crop': 'Cotton',
                'suitability': self._calculate_suitability(N, P, K, temperature, humidity, ph, rainfall,
                    ideal_N=(50, 100), ideal_P=(20, 50), ideal_K=(30, 60),
                    ideal_temp=(25, 35), ideal_humidity=(50, 70), ideal_ph=(5.5, 8.0), ideal_rainfall=(50, 100)),
                'season': 'Kharif',
                'roi': 38
            },
            {
                'crop': 'Sugarcane',
                'suitability': self._calculate_suitability(N, P, K, temperature, humidity, ph, rainfall,
                    ideal_N=(100, 200), ideal_P=(30, 60), ideal_K=(50, 100),
                    ideal_temp=(26, 32), ideal_humidity=(60, 80), ideal_ph=(6.0, 7.5), ideal_rainfall=(100, 200)),
                'season': 'Year-round',
                'roi': 42
            },
            {
                'crop': 'Maize',
                'suitability': self._calculate_suitability(N, P, K, temperature, humidity, ph, rainfall,
                    ideal_N=(50, 120), ideal_P=(15, 40), ideal_K=(20, 50),
                    ideal_temp=(18, 27), ideal_humidity=(50, 70), ideal_ph=(5.5, 7.0), ideal_rainfall=(50, 150)),
                'season': 'Kharif',
                'roi': 35
            },
            {
                'crop': 'Groundnut',
                'suitability': self._calculate_suitability(N, P, K, temperature, humidity, ph, rainfall,
                    ideal_N=(20, 50), ideal_P=(10, 30), ideal_K=(20, 40),
                    ideal_temp=(25, 30), ideal_humidity=(50, 70), ideal_ph=(6.0, 7.0), ideal_rainfall=(50, 125)),
                'season': 'Kharif',
                'roi': 32
            },
            {
                'crop': 'Soybean',
                'suitability': self._calculate_suitability(N, P, K, temperature, humidity, ph, rainfall,
                    ideal_N=(20, 60), ideal_P=(15, 40), ideal_K=(20, 50),
                    ideal_temp=(20, 30), ideal_humidity=(60, 80), ideal_ph=(6.0, 7.0), ideal_rainfall=(60, 100)),
                'season': 'Kharif',
                'roi': 30
            }
        ]
        
        # Sort by suitability
        crop_rules.sort(key=lambda x: x['suitability'], reverse=True)
        
        # Get top 5
        for crop_data in crop_rules[:5]:
            recommendations.append({
                'crop': crop_data['crop'],
                'suitability': round(crop_data['suitability'], 2),
                'season': crop_data['season'],
                'roi': crop_data['roi'],
                'confidence': round(crop_data['suitability'], 2)
            })
        
        return {
            'success': True,
            'primary_crop': recommendations[0]['crop'] if recommendations else 'Rice',
            'recommendations': recommendations,
            'method': 'rule_based'
        }
    
    def _calculate_suitability(self, N, P, K, temp, humidity, ph, rainfall,
                              ideal_N, ideal_P, ideal_K, ideal_temp, ideal_humidity, ideal_ph, ideal_rainfall):
        """Calculate suitability score based on how close values are to ideal ranges"""
        score = 0
        total_weight = 0
        
        # Nitrogen (weight: 0.15)
        if ideal_N[0] <= N <= ideal_N[1]:
            score += 100 * 0.15
        else:
            distance = min(abs(N - ideal_N[0]), abs(N - ideal_N[1]))
            score += max(0, 100 - (distance / ideal_N[1]) * 100) * 0.15
        total_weight += 0.15
        
        # Phosphorus (weight: 0.15)
        if ideal_P[0] <= P <= ideal_P[1]:
            score += 100 * 0.15
        else:
            distance = min(abs(P - ideal_P[0]), abs(P - ideal_P[1]))
            score += max(0, 100 - (distance / ideal_P[1]) * 100) * 0.15
        total_weight += 0.15
        
        # Potassium (weight: 0.15)
        if ideal_K[0] <= K <= ideal_K[1]:
            score += 100 * 0.15
        else:
            distance = min(abs(K - ideal_K[0]), abs(K - ideal_K[1]))
            score += max(0, 100 - (distance / ideal_K[1]) * 100) * 0.15
        total_weight += 0.15
        
        # Temperature (weight: 0.20)
        if ideal_temp[0] <= temp <= ideal_temp[1]:
            score += 100 * 0.20
        else:
            distance = min(abs(temp - ideal_temp[0]), abs(temp - ideal_temp[1]))
            score += max(0, 100 - (distance / ideal_temp[1]) * 100) * 0.20
        total_weight += 0.20
        
        # Humidity (weight: 0.10)
        if ideal_humidity[0] <= humidity <= ideal_humidity[1]:
            score += 100 * 0.10
        else:
            distance = min(abs(humidity - ideal_humidity[0]), abs(humidity - ideal_humidity[1]))
            score += max(0, 100 - (distance / ideal_humidity[1]) * 100) * 0.10
        total_weight += 0.10
        
        # pH (weight: 0.15)
        if ideal_ph[0] <= ph <= ideal_ph[1]:
            score += 100 * 0.15
        else:
            distance = min(abs(ph - ideal_ph[0]), abs(ph - ideal_ph[1]))
            score += max(0, 100 - (distance / ideal_ph[1]) * 100) * 0.15
        total_weight += 0.15
        
        # Rainfall (weight: 0.10)
        if ideal_rainfall[0] <= rainfall <= ideal_rainfall[1]:
            score += 100 * 0.10
        else:
            distance = min(abs(rainfall - ideal_rainfall[0]), abs(rainfall - ideal_rainfall[1]))
            score += max(0, 100 - (distance / ideal_rainfall[1]) * 100) * 0.10
        total_weight += 0.10
        
        return score / total_weight if total_weight > 0 else 0


# Global instance
_recommender = None

def get_recommender():
    """Get or create the global recommender instance"""
    global _recommender
    if _recommender is None:
        # Try to find model file (prioritize XGBoost.pkl from notebook)
        model_path = None
        for path in ['XGBoost.pkl', 'crop_model.pkl', 'XGBoost_model.pkl']:
            if os.path.exists(path):
                model_path = path
                print(f"Found model file: {path}")
                break
        
        # Try to find CSV file for training if model not found
        csv_path = None
        if not model_path:
            for path in ['Crop_recommendation.csv', 'crop_recommendation.csv', '/content/Crop_recommendation.csv']:
                if os.path.exists(path):
                    csv_path = path
                    break
        
        _recommender = CropRecommender(
            model_path=model_path or 'crop_model.pkl',
            csv_path=csv_path
        )
    return _recommender

