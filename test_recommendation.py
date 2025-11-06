"""
Quick test script for MVP testing of the crop recommendation endpoint
"""

import requests
import json

def test_recommendation_endpoint():
    """Test the /recommend endpoint with sample data"""
    
    # Test data - typical values for Tamil Nadu
    test_data = {
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
    
    print("=" * 80)
    print("🧪 TESTING CROP RECOMMENDATION ENDPOINT")
    print("=" * 80)
    print("\n📤 Sending request to /recommend endpoint...")
    print(f"Request data:\n{json.dumps(test_data, indent=2)}")
    print("\n" + "-" * 80 + "\n")
    
    try:
        # Make request to Flask server
        response = requests.post(
            'http://127.0.0.1:5000/recommend',
            json=test_data,
            headers={'Content-Type': 'application/json'}
        )
        
        print(f"📥 Response Status: {response.status_code}")
        print(f"📥 Response Headers: {dict(response.headers)}")
        print("\n" + "-" * 80 + "\n")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ SUCCESS! Recommendations received:")
            print(json.dumps(result, indent=2))
            
            if result.get('success'):
                print("\n" + "=" * 80)
                print("🌾 TOP RECOMMENDATIONS:")
                print("=" * 80)
                for i, rec in enumerate(result.get('recommendations', [])[:5], 1):
                    print(f"\n{i}. {rec['crop']}")
                    print(f"   Suitability: {rec['suitability']}%")
                    print(f"   Season: {rec.get('season', 'N/A')}")
                    print(f"   ROI: {rec.get('roi', 'N/A')}%")
                    print(f"   Reason: {rec.get('reason', 'N/A')}")
                print("\n" + "=" * 80)
                print(f"✓ Method used: {result.get('method', 'unknown')}")
                print(f"✓ Primary crop: {result.get('primary_crop', 'N/A')}")
                return True
            else:
                print(f"❌ Error: {result.get('error', 'Unknown error')}")
                return False
        else:
            print(f"❌ ERROR: Status code {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ ERROR: Could not connect to Flask server.")
        print("   Make sure the server is running: python app.py")
        return False
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    print("\n🚀 Starting MVP Test...\n")
    success = test_recommendation_endpoint()
    
    if success:
        print("\n✅ MVP TEST PASSED!")
    else:
        print("\n❌ MVP TEST FAILED!")
        print("\nTroubleshooting:")
        print("1. Make sure Flask server is running: python app.py")
        print("2. Check that all dependencies are installed: pip install -r requirements.txt")
        print("3. Verify the endpoint is accessible at http://127.0.0.1:5000/recommend")

