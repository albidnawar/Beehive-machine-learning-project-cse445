# hive_health.py

# Define ideal conditions and weights for each feature
OPTIMAL_VALUES = {
    'hive_temperature': 35.0,  # Ideal temperature in °C
    'hive_humidity': 50.0,     # Ideal humidity in %
    'hive_pressure': 1013.0,   # Standard atmospheric pressure in hPa
    'weather_temperature': 25.0,  # Ideal weather temperature in °C
    'weather_humidity': 50.0    # Ideal weather humidity in %
}

# Weight values for the features, based on importance (these are just examples, adjust as needed)
WEIGHTS = {
    'hive_temperature': 1,
    'hive_humidity': 1,
    'hive_pressure': 1,
    'weather_temperature': 1,
    'weather_humidity': 1,
}

# Function to calculate the normalized score for each feature
def calculate_score(actual_value, optimal_value):
    return 1 - abs(actual_value - optimal_value) / optimal_value

# Function to compute Hive Health Score
def compute_hive_health(extracted_data):
    # Calculate individual feature scores
    hive_temp_score = calculate_score(extracted_data['hive_temperature'], OPTIMAL_VALUES['hive_temperature'])
    hive_humidity_score = calculate_score(extracted_data['hive_humidity'], OPTIMAL_VALUES['hive_humidity'])
    hive_pressure_score = calculate_score(extracted_data['hive_pressure'], OPTIMAL_VALUES['hive_pressure'])
    weather_temp_score = calculate_score(extracted_data['weather_temperature'], OPTIMAL_VALUES['weather_temperature'])
    weather_humidity_score = calculate_score(extracted_data['weather_humidity'], OPTIMAL_VALUES['weather_humidity'])
    
    # Compute the weighted average for Hive Health Score
    weighted_sum = (WEIGHTS['hive_temperature'] * hive_temp_score +
                    WEIGHTS['hive_humidity'] * hive_humidity_score +
                    WEIGHTS['hive_pressure'] * hive_pressure_score +
                    WEIGHTS['weather_temperature'] * weather_temp_score +
                    WEIGHTS['weather_humidity'] * weather_humidity_score)
    
    total_weight = sum(WEIGHTS.values())
    
    hive_health_score = weighted_sum / total_weight
    
    # Classify the hive health score into categories
    if hive_health_score >= 0.9:
        health_category = "Good"
    elif hive_health_score >= 0.7:
        health_category = "Moderate"
    else:
        health_category = "Poor"
    
    return hive_health_score * 100, health_category
