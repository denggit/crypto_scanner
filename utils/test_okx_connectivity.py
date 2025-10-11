"""
Test OKX API connectivity
"""

import sys
import os
import requests

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from logger import logger

def test_connectivity():
    """Test basic connectivity to OKX API"""
    try:
        logger.info("Testing connectivity to OKX API...")
        response = requests.get("https://www.okx.com/api/v5/public/time", timeout=5)
        if response.status_code == 200:
            data = response.json()
            logger.info(f"[PASS] OKX API is accessible. Server time: {data}")
            return True
        else:
            logger.error(f"[FAIL] OKX API returned status code: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"[FAIL] Connectivity test failed: {e}")
        return False

def test_public_endpoints():
    """Test public endpoints"""
    try:
        logger.info("\nTesting public endpoints...")

        # Test time endpoint
        response = requests.get("https://www.okx.com/api/v5/public/time", timeout=5)
        if response.status_code == 200:
            logger.info("[PASS] Time endpoint accessible")
        else:
            logger.error(f"[FAIL] Time endpoint returned status code: {response.status_code}")

        # Test tickers endpoint
        response = requests.get("https://www.okx.com/api/v5/market/tickers?instType=SPOT", timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get('code') == '0':
                logger.info(f"[PASS] Tickers endpoint accessible. Retrieved {len(data.get('data', []))} tickers")
            else:
                logger.error(f"[FAIL] Tickers endpoint returned error: {data.get('msg')}")
        else:
            logger.error(f"[FAIL] Tickers endpoint returned status code: {response.status_code}")

        return True
    except Exception as e:
        logger.error(f"[FAIL] Public endpoints test failed: {e}")
        return False

def main():
    logger.info("OKX API Connectivity Test")
    logger.info("=" * 30)

    connectivity_ok = test_connectivity()
    endpoints_ok = test_public_endpoints()

    logger.info("\n" + "=" * 30)
    if connectivity_ok and endpoints_ok:
        logger.info("[SUCCESS] All connectivity tests passed!")
        return 0
    else:
        logger.error("[ERROR] Some connectivity tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())