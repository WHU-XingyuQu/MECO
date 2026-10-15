"""Input-condition routing used by the current cascade."""
def requires_gpu(observation):
    valid = observation.get("sensor_valid")
    if type(valid) is not bool:
        raise ValueError("difficulty routing requires an explicit sensor_valid boolean")
    return valid
