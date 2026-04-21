from pathlib import Path

FAILURE_TYPES = {
    'engine': 'engine_failure',
    'elevator': 'elevator_failure',
    'aileron': 'aileron_failure',
    'aileron_left': 'left_aileron',
    'aileron_right': 'right_aileron',
    'rudder': 'rudder',
    'rudder_left': 'rudder_left_failure',
    'rudder_right': 'rudder_right_failure',
    'no_failure': 'no_failure'
}

def extract_failure_type(flight_folder_name):
    name_lower = flight_folder_name.lower()
    
    if 'no_ground_truth' in name_lower:
        return 'no_ground_truth'
    elif 'no_failure' in name_lower:
        return 'no_failure'
    elif 'engine_failure' in name_lower:
        return 'engine'
    elif 'elevator_failure' in name_lower:
        return 'elevator'
    elif 'left_aileron' in name_lower or 'right_aileron' in name_lower:
        return 'aileron'
    elif 'rudder_left' in name_lower:
        return 'rudder_left'
    elif 'rudder_right' in name_lower:
        return 'rudder_right'
    elif 'rudder' in name_lower:
        return 'rudder'
    else:
        return 'unknown'

def filter_by_failure_type(flight_folders, failure_types=None):
    if failure_types is None:
        return flight_folders
    
    if isinstance(failure_types, str):
        failure_types = [failure_types]
    
    filtered = []
    for folder in flight_folders:
        failure_type = extract_failure_type(folder.name)
        if failure_type in failure_types:
            filtered.append(folder)
    
    return filtered

def get_failure_distribution(flight_folders):
    distribution = {}
    for folder in flight_folders:
        failure_type = extract_failure_type(folder.name)
        distribution[failure_type] = distribution.get(failure_type, 0) + 1
    return distribution
