import numpy as np

def load_and_preprocess(file_path, sequence_length=30):
    raw_data = np.loadtxt(file_path)
    
    features = raw_data[:, 2:]
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    std[std == 0] = 1.0
    normalized = (features - mean) / std
    
    engine_ids = raw_data[:, 0]
    unique_engines = np.unique(engine_ids)
    
    X_windows = []
    y_rul = []
    
    for engine in unique_engines:
        mask = (engine_ids == engine)
        engine_data = normalized[mask]
    
        engine_cycles = raw_data[mask, 1]
        max_cycles = np.max(engine_cycles)

        RUL = max_cycles - engine_cycles
        total_cycles = len(engine_data)
        
        for i in range(total_cycles - sequence_length + 1):
            window = engine_data[i : i + sequence_length]
            X_windows.append(window)
            y_rul.append(RUL[i + sequence_length - 1])
    
    return np.array(X_windows), np.array(y_rul), mean, std