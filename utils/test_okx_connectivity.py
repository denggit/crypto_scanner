"""
Test OKX API connectivity
"""

import sys
import os
import requests

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_connectivity():
    """Test basic connectivity to OKX API"""
    try:
        print("Testing connectivity to OKX API...")
        response = requests.get("https://www.okx.com/api/v5/public/time", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"[PASS] OKX API is accessible. Server time: {data}")
            return True
        else:
            print(f"[FAIL] OKX API returned status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"[FAIL] Connectivity test failed: {e}")
        return False

def test_public_endpoints():
    """Test public endpoints"""
    try:
        print("\nTesting public endpoints...")

        # Test time endpoint
        response = requests.get("https://www.okx.com/api/v5/public/time", timeout=5)
        if response.status_code == 200:
            print("[PASS] Time endpoint accessible")
        else:
            print(f"[FAIL] Time endpoint returned status code: {response.status_code}")

        # Test tickers endpoint
        response = requests.get("https://www.okx.com/api/v5/market/tickers?instType=SPOT", timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get('code') == '0':
                print(f"[PASS] Tickers endpoint accessible. Retrieved {len(data.get('data', []))} tickers")
            else:
                print(f"[FAIL] Tickers endpoint returned error: {data.get('msg')}")
        else:
            print(f"[FAIL] Tickers endpoint returned status code: {response.status_code}")

        return True
    except Exception as e:
        print(f"[FAIL] Public endpoints test failed: {e}")
        return False

def main():
    print("OKX API Connectivity Test")
    print("=" * 30)

    connectivity_ok = test_connectivity()
    endpoints_ok = test_public_endpoints()

    print("\n" + "=" * 30)
    if connectivity_ok and endpoints_ok:
        print("[SUCCESS] All connectivity tests passed!")
        return 0
    else:
        print("[ERROR] Some connectivity tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())