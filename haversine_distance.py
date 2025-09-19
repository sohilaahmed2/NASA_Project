import numpy as np

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculates the great circle distance between two points on Earth using the Haversine formula.
    
    Args:
        lat1 (float): Latitude of first point (degrees).
        lon1 (float): Longitude of first point (degrees).
        lat2 (float): Latitude of second point (degrees).
        lon2 (float): Longitude of second point (degrees).
    
    Returns:
        float: Distance in kilometers.
    
    Raises:
        ValueError: If coordinates are invalid (e.g., out of range).
    """
    try:
        # Validate coordinates
        if not (-90 <= lat1 <= 90 and -90 <= lat2 <= 90 and -180 <= lon1 <= 180 and -180 <= lon2 <= 180):
            raise ValueError("الإحداثيات غير صالحة: يجب أن تكون خطوط العرض بين -90 و90، وخطوط الطول بين -180 و180")
        
        # Earth's radius in km
        R = 6371
        
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        distance = R * c
        
        return distance
    except Exception as e:
        raise ValueError(f"خطأ في حساب المسافة: {str(e)}")