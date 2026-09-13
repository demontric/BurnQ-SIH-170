import pandas as pd
import numpy as np
import os
import random

def generate_synthetic_data(num_lots=5, random_seed=42):
    np.random.seed(random_seed)
    random.seed(random_seed)

    parameters = {
        'leakage_current': {'baseline': 10, 'noise': 2, 'limit': 50, 'tau': 30},
        'iddq': {'baseline': 100, 'noise': 15, 'limit': 300, 'tau': 40},
        'prop_delay': {'baseline': 5, 'noise': 0.5, 'limit': 10, 'tau': 50}
    }

    data = []
    part_counter = 1

    for lot_id in range(1, num_lots + 1):
        num_parts = random.randint(50, 200)
        
        for param, props in parameters.items():
            baseline = props['baseline']
            noise_std = props['noise']
            limit = props['limit']
            tau = props['tau']
            
            # 3-8% defect rate
            num_defects = int(num_parts * random.uniform(0.03, 0.08))
            defect_indices = set(random.sample(range(num_parts), num_defects))

            for i in range(num_parts):
                part_id = f"P_{lot_id:03d}_{i:03d}"
                is_defect = i in defect_indices

                # Saturation curve: V(t) = V0 + a*(1 - exp(-t/tau))
                # where a is the total drift
                v0 = np.random.normal(baseline, noise_std)
                a = np.random.normal(baseline * 0.2, noise_std * 0.5)

                if is_defect:
                    defect_type = random.choice(['absolute', 'drift'])
                    if defect_type == 'absolute':
                        # Absolute outlier but within limit
                        # Shift the baseline up significantly but cap below limit
                        shift = (limit - baseline) * random.uniform(0.7, 0.95)
                        v0 = baseline + shift
                        v0 = min(v0, limit - 0.1) # ensure it's strictly within limit
                    else:
                        # Abnormal drift: steep rise instead of saturation
                        # Linear or exponential drift that exceeds normal saturation, but stays just under limit
                        a = np.random.normal(baseline * 1.5, noise_std) 
                        tau = tau * 5 # much slower saturation = looks like linear rise

                def get_val(t):
                    val = v0 + a * (1 - np.exp(-t / tau))
                    val += np.random.normal(0, noise_std * 0.1) # measurement noise
                    return val

                val_0h = get_val(0)
                val_24h = get_val(24)
                val_96h = get_val(96)
                val_168h = get_val(168)
                
                # Make sure it doesn't exceed datasheet limit for this synthetic exercise
                # If it exceeds naturally, we just clamp it slightly below for the sake of 'latent' defect simulation
                if val_168h > limit:
                     val_168h = limit - np.random.uniform(0.1, 0.5)
                if val_96h > val_168h:
                     val_96h = val_168h - np.random.uniform(0.1, 0.5)
                if val_24h > val_96h:
                     val_24h = val_96h - np.random.uniform(0.1, 0.5)
                if val_0h > val_24h:
                     val_0h = val_24h - np.random.uniform(0.1, 0.5)

                data.append({
                    'part_id': part_id,
                    'lot_id': f"LOT_{lot_id:03d}",
                    'parameter': param,
                    'value_0h': val_0h,
                    'value_24h': val_24h,
                    'value_96h': val_96h,
                    'value_168h': val_168h,
                    'datasheet_limit': limit,
                    'ground_truth_defective': is_defect
                })

    df = pd.DataFrame(data)
    
    # Save to data directory
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, 'synthetic_burnin_data.csv')
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} rows of synthetic data at {output_path}")
    return df

if __name__ == "__main__":
    generate_synthetic_data()
