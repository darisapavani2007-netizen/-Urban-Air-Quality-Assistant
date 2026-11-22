# src/data_clients/airnow_client.py
import os
from typing import Dict, Any, Optional
import logging

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    httpx = None
    HTTPX_AVAILABLE = False

from src.data_clients.base_client import BaseDataClient

logger = logging.getLogger(__name__)


class AirNowClient(BaseDataClient):
    """Client for AirNow API with optional API key"""

    def __init__(self):
        super().__init__()
        self.api_key = os.getenv("AIRNOW_API_KEY")
        self.base_url = "http://www.airnowapi.org/aq/"

    async def get_initial_data(self):
        """Initial data documents if needed"""
        return []

    async def health_check(self) -> bool:
        """Check if API is reachable"""
        if not self.api_key or not HTTPX_AVAILABLE:
            return True  # Optional service
            
        try:
            url = f"{self.base_url}forecast/zipCode/"
            params = {
                "format": "application/json",
                "zipCode": "20002",
                "date": "2025-10-02",
                "distance": "25",
                "API_KEY": self.api_key
            }
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.get(url, params=params)
                return response.status_code == 200
        except Exception as e:
            logger.warning(f"AirNow health check: {e}")
            return True

    async def get_current_air_quality(self, lat: float, lon: float) -> Dict[str, Any]:
        """Fetch current AQI for a given lat/lon"""
        if not self.api_key or not HTTPX_AVAILABLE:
            return {"error": "AirNow API key not configured"}
            
        try:
            url = f"{self.base_url}observation/latLong/current/"
            params = {
                "format": "application/json",
                "latitude": lat,
                "longitude": lon,
                "distance": 25,
                "API_KEY": self.api_key
            }
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                return response.json()
        except Exception as e:
            return {"error": str(e)}
