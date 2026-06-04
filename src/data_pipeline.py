import numpy as np

def load_and_preprocess(file_path, sequence_length=30, mode='train', mean=None, std=None):
    """
    Unified Data Pipeline for NASA Turbofan Telemetry.
    Modes:
        'train': Returns all overlapping sliding windows across all timelines.
        'test': Returns ONLY the absolute last historical window per engine.
    """
    raw_data = np.loadtxt(file_path)
    engine_ids = raw_data[:, 0]
    features = raw_data[:, 2:]
    
    # Compute or apply scaling factors to completely stop the Data Leakage
    if mode == 'train':
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0)
        std[std == 0] = 1.0
    elif mode == 'test':
        if mean is None or std is None:
            raise ValueError("[Pipeline Error] Test mode requires training mean and std constants.")
            
    normalized = (features - mean) / std
    unique_engines = np.unique(engine_ids)
    
    X_windows = []
    y_indicators = [] # Tracks either intermediate cycles or the engine pointer
    
    for engine in unique_engines:
        mask = (engine_ids == engine)
        engine_data = normalized[mask]
        engine_cycles = raw_data[mask, 1]
        
        if len(engine_data) < sequence_length:
            continue
            
        if mode == 'train':
            # Extract all sequential slices
            max_cycles = np.max(engine_cycles)
            RUL = max_cycles - engine_cycles
            total_cycles = len(engine_data)
            
            for i in range(total_cycles - sequence_length + 1):
                X_windows.append(engine_data[i : i + sequence_length])
                y_indicators.append(RUL[i + sequence_length - 1])
                
        elif mode == 'test':
            # Structural Pattern: Extract ONLY the final state window for inference
            X_windows.append(engine_data[-sequence_length:])
            # Append the engine index to align perfectly with RUL_FD001.txt
            y_indicators.append(engine)
            
    return np.array(X_windows), np.array(y_indicators), mean, std