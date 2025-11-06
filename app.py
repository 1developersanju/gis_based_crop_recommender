# First, install Flask if not already installed:
# pip install flask requests

from flask import Flask, request, jsonify, send_file
import requests
import os
import json
from crop_recommender import get_recommender

app = Flask(__name__)

@app.route('/')
def index():
    return send_file('index.html')

@app.route('/proxy', methods=['POST'])
def proxy_land_details():
    # Get the form data from the request
    lat = request.form.get('latitude')
    lng = request.form.get('longitude')
    up = request.form.get('up', '')

    print("\n" + "="*80)
    print("📍 LOCATION TAPPED - Request Received")
    print("="*80)
    print(f"Latitude:  {lat}")
    print(f"Longitude: {lng}")
    print(f"UP:        {up}")
    print("-"*80)

    if not lat or not lng:
        print("❌ ERROR: Missing latitude or longitude")
        return jsonify({'error': 'Missing latitude or longitude'}), 400

    # Target URL
    target_url = 'https://tngis.tn.gov.in/apps/generic_api/v2/land_details'

    # Headers to mimic the original request (adjust as needed)
    headers = {
        'Accept': '*/*',
        'Accept-Language': 'en-US,en;q=0.7',
        'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
        'Sec-Ch-Ua': '"Not;A=Brand";v="99", "Brave";v="139", "Chromium";v="139"',
        'Sec-Ch-Ua-Mobile': '?0',
        'Sec-Ch-Ua-Platform': '"macOS"',
        'Sec-Fetch-Dest': 'empty',
        'Sec-Fetch-Mode': 'cors',
        'Sec-Fetch-Site': 'same-origin',
        'Sec-Gpc': '1',
        'X-App-Name': 'demo',
        'X-Requested-With': 'XMLHttpRequest',
        'Referer': 'https://tngis.tn.gov.in/apps/gi_viewer/',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36'
    }

    # Body
    body = f'latitude={lat}&longitude={lng}&up={up}'
    print(f"Request URL: {target_url}")
    print(f"Request Body: {body}")

    try:
        # Forward the request
        print("\n🔄 Sending request to external API...")
        response = requests.post(target_url, headers=headers, data=body)
        
        # Print all response details
        print("\n" + "="*80)
        print("📥 COMPLETE RESPONSE FROM EXTERNAL API")
        print("="*80)
        
        # Status Code
        print(f"\n📊 Status Code: {response.status_code}")
        print(f"📊 Status Reason: {response.reason}")
        
        # Response Headers
        print("\n📋 Response Headers:")
        print("-"*80)
        for header_name, header_value in response.headers.items():
            print(f"  {header_name}: {header_value}")
        print("-"*80)
        
        # Raw Response Text (before parsing)
        print("\n📄 Raw Response Text (first 1000 chars):")
        print("-"*80)
        raw_text = response.text
        print(raw_text[:1000])
        if len(raw_text) > 1000:
            print(f"\n... (truncated, total length: {len(raw_text)} characters)")
        print("-"*80)
        
        # Full Response Text
        print("\n📄 Full Raw Response Text:")
        print("-"*80)
        print(raw_text)
        print("-"*80)
        
        response.raise_for_status()
        
        # Parse response
        try:
            response_data = response.json()
            
            print("\n✅ Parsed JSON Response:")
            print("-"*80)
            print(json.dumps(response_data, indent=2, ensure_ascii=False))
            print("-"*80)
            
            # Detailed analysis of response structure
            print("\n🔍 Response Structure Analysis:")
            print("-"*80)
            
            if isinstance(response_data, dict):
                print(f"✓ Response is a dictionary with {len(response_data)} keys")
                print(f"  Keys: {list(response_data.keys())}")
                
                # Check success flag
                if 'success' in response_data:
                    success_val = response_data.get('success')
                    print(f"✓ Success field: {success_val} (type: {type(success_val).__name__})")
                    
                    if success_val == 1:
                        print("  → Success flag indicates success (1)")
                    else:
                        print(f"  → Success flag indicates failure/non-success ({success_val})")
                else:
                    print("⚠ Success field: Missing")
                
                # Check data object
                if 'data' in response_data:
                    data = response_data['data']
                    print(f"✓ Data object: Present (type: {type(data).__name__})")
                    
                    if isinstance(data, dict):
                        print(f"  → Data has {len(data)} fields")
                        print(f"  → Data keys: {list(data.keys())}")
                        
                        # Check for specific fields
                        if 'geojson_geom' in data:
                            geojson_str = data['geojson_geom']
                            print(f"✓ GeoJSON geometry: Present")
                            print(f"  → Type: {type(geojson_str).__name__}")
                            if isinstance(geojson_str, str):
                                print(f"  → Length: {len(geojson_str)} characters")
                                try:
                                    geojson_parsed = json.loads(geojson_str)
                                    print(f"  → Valid JSON: Yes")
                                    if isinstance(geojson_parsed, dict):
                                        print(f"  → GeoJSON type: {geojson_parsed.get('type', 'N/A')}")
                                        if geojson_parsed.get('type') == 'Polygon':
                                            coords = geojson_parsed.get('coordinates', [])
                                            if coords and len(coords) > 0:
                                                print(f"  → Coordinate points: {len(coords[0])}")
                                except:
                                    print(f"  → Valid JSON: No (not parseable)")
                        else:
                            print("⚠ GeoJSON geometry: Missing")
                        
                        # Print all data fields
                        print("\n  📝 All Data Fields:")
                        for key, value in data.items():
                            value_type = type(value).__name__
                            if isinstance(value, str):
                                value_preview = value[:100] + "..." if len(value) > 100 else value
                                print(f"    {key}: {value_preview} (type: {value_type}, length: {len(value)})")
                            else:
                                print(f"    {key}: {value} (type: {value_type})")
                    else:
                        print(f"  → Data is not a dictionary: {type(data).__name__}")
                        print(f"  → Data value: {data}")
                else:
                    print("⚠ Data object: Missing")
                
                # Print all top-level fields
                print("\n📝 All Top-Level Response Fields:")
                for key, value in response_data.items():
                    if key != 'data':  # Already handled above
                        value_type = type(value).__name__
                        if isinstance(value, (dict, list)):
                            print(f"  {key}: {json.dumps(value, indent=4, ensure_ascii=False)[:200]}... (type: {value_type})")
                        elif isinstance(value, str) and len(value) > 100:
                            print(f"  {key}: {value[:100]}... (type: {value_type}, length: {len(value)})")
                        else:
                            print(f"  {key}: {value} (type: {value_type})")
            elif isinstance(response_data, list):
                print(f"✓ Response is a list with {len(response_data)} items")
                print(f"  First item: {json.dumps(response_data[0] if response_data else None, indent=2)[:500]}")
            else:
                print(f"✓ Response type: {type(response_data).__name__}")
                print(f"  Value: {response_data}")
            
            print("-"*80)
            
        except json.JSONDecodeError as json_err:
            print(f"\n❌ JSON Parse Error: {str(json_err)}")
            print("  Raw response is not valid JSON")
            print("-"*80)
            # Return error but still print what we got
            response_data = {'error': 'Invalid JSON response', 'raw_response': raw_text}
            
        # Create Flask response
        if 'response_data' not in locals():
            response_data = {'raw_response': raw_text}
            
        flask_response = jsonify(response_data)
        flask_response.headers.add('Access-Control-Allow-Origin', '*')
        flask_response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        flask_response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        
        print("="*80 + "\n")
        return flask_response
        
    except requests.exceptions.RequestException as e:
        print(f"\n❌ REQUEST ERROR: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response status: {e.response.status_code}")
            print(f"Response text: {e.response.text[:500]}")
        print("="*80 + "\n")
        return jsonify({'error': str(e)}), 500
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        print("Traceback:")
        traceback.print_exc()
        print("="*80 + "\n")
        return jsonify({'error': str(e)}), 500

# Handle preflight OPTIONS request for CORS
@app.route('/proxy', methods=['OPTIONS'])
def proxy_options():
    response = jsonify({})
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    return response

# Soil Card API endpoint
@app.route('/soil-card', methods=['POST'])
def get_soil_card():
    """Fetch soil card data from TN Agrisnet"""
    try:
        district = request.form.get('district', '')
        block = request.form.get('block', '')
        village = request.form.get('village', '')
        survey_no = request.form.get('survey_no', '')
        
        if not all([district, block, village, survey_no]):
            return jsonify({
                'success': False,
                'error': 'Missing required parameters: district, block, village, survey_no'
            }), 400
        
        # Call TN Agrisnet API
        target_url = 'https://www.tnagrisnet.tn.gov.in/mannvalam/soilCardPublic/en'
        headers = {
            'Content-Type': 'application/x-www-form-urlencoded',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Referer': 'https://www.tnagrisnet.tn.gov.in/mannvalam/soilCardPublic/en'
        }
        
        data = {
            'district': district,
            'block': block,
            'village': village,
            'Survey_no': survey_no,
            'ownerName': '',
            'mobileNumber': ''
        }
        
        response = requests.post(target_url, headers=headers, data=data)
        response.raise_for_status()
        
        # Parse HTML response (simplified - you may want to use BeautifulSoup for better parsing)
        # For now, return a placeholder structure
        return jsonify({
            'success': True,
            'soil': {
                'ph': 7.2,
                'nitrogen': 150,
                'phosphorus': 25,
                'potassium': 180,
                'moisture': 45
            }
        })
        
    except Exception as e:
        print(f"Error fetching soil card: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Average soil data endpoint (fallback)
@app.route('/soil-avg', methods=['GET'])
def get_soil_avg():
    """Get average soil data for a taluk/district"""
    taluk = request.args.get('taluk', '')
    
    # Return average values (placeholder)
    return jsonify({
        'success': True,
        'avg': {
            'ph': 7.0,
            'nitrogen': 140,
            'phosphorus': 22,
            'potassium': 170,
            'moisture': 40
        }
    })

# Climate data endpoint
@app.route('/climate', methods=['GET'])
def get_climate():
    """Fetch climate data for given coordinates"""
    try:
        lat = request.args.get('lat')
        lng = request.args.get('lng')
        
        if not lat or not lng:
            return jsonify({
                'success': False,
                'error': 'Missing lat/lng parameters'
            }), 400
        
        # Use Open-Meteo API or similar (placeholder for now)
        # You can integrate with Open-Meteo: https://open-meteo.com/en/docs
        return jsonify({
            'success': True,
            'climate': {
                'temperature_2m_max': 32,
                'temperature_2m_min': 24,
                'rainfall': 1200
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Crop recommendations endpoint
@app.route('/recommend', methods=['POST'])
def get_recommendations():
    """Get crop recommendations based on soil and climate data"""
    try:
        data = request.get_json()
        
        # Extract parameters from request
        # Expected format: {soil: {ph, nitrogen, phosphorus, potassium, moisture}, climate: {temperature_2m_max, temperature_2m_min, rainfall}}
        
        # Get soil data
        soil = data.get('soil', {})
        climate = data.get('climate', {})
        
        # Extract values with defaults
        N = float(soil.get('nitrogen', soil.get('N', 50)))  # Nitrogen in ppm/kg
        P = float(soil.get('phosphorus', soil.get('P', 25)))  # Phosphorus in ppm/kg
        K = float(soil.get('potassium', soil.get('K', 30)))  # Potassium in ppm/kg
        ph = float(soil.get('ph', soil.get('pH', 7.0)))
        
        # Get temperature (average of max and min, or use max)
        temp_max = float(climate.get('temperature_2m_max', climate.get('temperature_max', 28)))
        temp_min = float(climate.get('temperature_2m_min', climate.get('temperature_min', 20)))
        temperature = (temp_max + temp_min) / 2
        
        # Get humidity (default if not provided)
        humidity = float(climate.get('humidity', climate.get('humidity_2m', 65)))
        
        # Get rainfall
        rainfall = float(climate.get('rainfall', climate.get('precipitation', 100)))
        
        print(f"\n{'='*80}")
        print("🌾 CROP RECOMMENDATION REQUEST")
        print(f"{'='*80}")
        print(f"Nitrogen (N):     {N} ppm")
        print(f"Phosphorus (P):   {P} ppm")
        print(f"Potassium (K):    {K} ppm")
        print(f"pH:               {ph}")
        print(f"Temperature:      {temperature}°C (max: {temp_max}, min: {temp_min})")
        print(f"Humidity:         {humidity}%")
        print(f"Rainfall:         {rainfall} mm")
        print(f"{'='*80}\n")
        
        # Get recommender instance
        recommender = get_recommender()
        
        # Get predictions
        result = recommender.predict(N, P, K, temperature, humidity, ph, rainfall)
        
        if result['success']:
            # Format recommendations for frontend
            formatted_recommendations = []
            for rec in result['recommendations']:
                formatted_rec = {
                    'crop': rec['crop'],
                    'suitability': rec['suitability'],
                    'confidence': rec.get('confidence', rec['suitability']),
                    'season': rec.get('season', 'Year-round'),
                    'roi': rec.get('roi', 35),
                    'reason': f"Based on {result['method']} analysis with {rec.get('confidence', rec['suitability']):.1f}% confidence"
                }
                formatted_recommendations.append(formatted_rec)
            
            print(f"✓ Recommendations generated using {result['method']}")
            print(f"  Primary crop: {result['primary_crop']}")
            print(f"  Total recommendations: {len(formatted_recommendations)}")
            print(f"{'='*80}\n")
            
            return jsonify({
                'success': True,
                'primary_crop': result['primary_crop'],
                'recommendations': formatted_recommendations,
                'method': result['method'],
                'input_parameters': {
                    'N': N,
                    'P': P,
                    'K': K,
                    'temperature': temperature,
                    'humidity': humidity,
                    'ph': ph,
                    'rainfall': rainfall
                }
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to generate recommendations'
            }), 500
        
    except KeyError as e:
        print(f"❌ Missing parameter: {e}")
        return jsonify({
            'success': False,
            'error': f'Missing required parameter: {str(e)}'
        }), 400
    except ValueError as e:
        print(f"❌ Invalid parameter value: {e}")
        return jsonify({
            'success': False,
            'error': f'Invalid parameter value: {str(e)}'
        }), 400
    except Exception as e:
        print(f"❌ Error generating recommendations: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# CORS headers for all endpoints
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

if __name__ == '__main__':
    app.run(port=5000, debug=True)