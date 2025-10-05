"""
Smart Data Processor for NASA Kepler and TESS Data
Handles variable comment lines and different column formats
"""

import pandas as pd
import numpy as np

# Column mapping for Kepler ↔ TESS ↔ Standard names
COLUMN_MAPPING = {
    'kepler': {
        'period': 'koi_period',
        'depth': 'koi_depth',
        'duration': 'koi_duration',
        'planet_radius': 'koi_prad',
        'equilibrium_temp': 'koi_teq',
        'insolation_flux': 'koi_insol',
        'stellar_temp': 'koi_steff',
        'stellar_gravity': 'koi_slogg',
        'stellar_radius': 'koi_srad',
        'period_err1': 'koi_period_err1',
        'period_err2': 'koi_period_err2',
        'depth_err1': 'koi_depth_err1',
        'depth_err2': 'koi_depth_err2',
        'disposition': 'koi_disposition'
    },
    'tess': {
        'period': 'pl_orbper',
        'depth': 'pl_trandep',
        'duration': 'pl_trandurh',  # Note: 'h' for hours
        'planet_radius': 'pl_rade',
        'equilibrium_temp': 'pl_eqt',
        'insolation_flux': 'pl_insol',
        'stellar_temp': 'st_teff',
        'stellar_gravity': 'st_logg',
        'stellar_radius': 'st_rad',
        'period_err1': 'pl_orbpererr1',
        'period_err2': 'pl_orbpererr2',
        'depth_err1': 'pl_trandeperr1',
        'depth_err2': 'pl_trandeperr2',
        'disposition': 'tfopwg_disp'
    }
}

def smart_csv_reader(file_path):
    """
    Intelligently read NASA CSV files
    - Auto-skips comment lines (starting with #)
    - Works with any number of comment lines
    - Returns DataFrame and metadata
    """
    # Read file to find data start
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    
    # Find first non-comment line
    data_start = 0
    for i, line in enumerate(lines):
        if not line.strip().startswith('#'):
            data_start = i
            break
    
    # Read CSV from data start
    df = pd.read_csv(file_path, skiprows=data_start, low_memory=False)
    
    return df, data_start


def detect_mission_type(columns):
    """
    Auto-detect if data is from Kepler or TESS based on column names
    
    Returns: 'kepler', 'tess', or None
    """
    columns_str = '|'.join([str(col).lower() for col in columns])
    
    # Check for Kepler-specific columns
    if 'koi_' in columns_str:
        return 'kepler'
    
    # Check for TESS-specific columns
    elif any(col in columns_str for col in ['pl_orbper', 'st_teff', 'tfopwg']):
        return 'tess'
    
    else:
        return None


def standardize_columns(df, mission_type):
    """
    Convert mission-specific column names to standard names
    Preserves star names and IDs for results display
    
    Returns: DataFrame with standardized column names, list of missing columns
    """
    if mission_type not in COLUMN_MAPPING:
        return None, f"Unknown mission type: {mission_type}"
    
    mapping = COLUMN_MAPPING[mission_type]
    
    # Check which standard columns we can create
    standardized = pd.DataFrame()
    missing = []
    
    for standard_name, original_name in mapping.items():
        if original_name in df.columns:
            standardized[standard_name] = df[original_name]
        else:
            missing.append(f"{standard_name} ({original_name})")
    
    # Add star names for display (mission-specific)
    if mission_type == 'kepler':
        # Prefer kepler_name, fallback to kepoi_name, fallback to kepid
        if 'kepler_name' in df.columns:
            standardized['star_name'] = df['kepler_name'].fillna(df.get('kepoi_name', 'Unknown'))
        elif 'kepoi_name' in df.columns:
            standardized['star_name'] = df['kepoi_name']
        else:
            standardized['star_name'] = 'Kepler-' + df['kepid'].astype(str)
        
        standardized['star_id'] = df.get('kepid', None)
        
    elif mission_type == 'tess':
        # Use toi for name, tid for ID
        if 'toi' in df.columns:
            standardized['star_name'] = 'TOI-' + df['toi'].astype(str)
        else:
            standardized['star_name'] = 'TESS-' + df['tid'].astype(str)
        
        standardized['star_id'] = df.get('tid', df.get('tic_id', None))
    
    return standardized, missing


def engineer_features(df):
    """
    Create all 22 features from base measurements
    Preserves disposition column for label creation
    Input: DataFrame with 9 base features + error columns
    Output: DataFrame with 22 features + disposition (if present)
    """
    result = df.copy()
    
    # Preserve disposition if it exists (for retraining)
    preserve_disposition = 'disposition' in df.columns
    
    # Basic engineered features (3)
    result['depth_per_duration'] = df['depth'] / (df['duration'] + 1e-10)
    result['planet_star_ratio'] = df['planet_radius'] / (df['stellar_radius'] + 1e-10)
    result['temp_insol_ratio'] = df['equilibrium_temp'] / (df['insolation_flux'] + 1e-10)
    
    # Advanced features using error columns (if available)
    if 'period_err1' in df.columns and 'period_err2' in df.columns:
        period_err_avg = (np.abs(df['period_err1']) + np.abs(df['period_err2'])) / 2
        result['period_snr'] = df['period'] / (period_err_avg + 1e-10)
    else:
        result['period_snr'] = df['period'] / 0.01  # Default small error
    
    if 'depth_err1' in df.columns and 'depth_err2' in df.columns:
        depth_err_avg = (np.abs(df['depth_err1']) + np.abs(df['depth_err2'])) / 2
        result['depth_snr'] = df['depth'] / (depth_err_avg + 1e-10)
    else:
        result['depth_snr'] = df['depth'] / 1.0  # Default small error
    
    # Physics-based features
    expected_depth = (df['planet_radius'] / (df['stellar_radius'] + 1e-10)) ** 2
    result['depth_consistency'] = df['depth'] / (expected_depth + 1e-10)
    
    result['orbital_speed'] = (2 * np.pi * df['stellar_radius']) / (df['period'] + 1e-10)
    result['semi_major_axis'] = df['period'] ** (2/3)
    result['stellar_density'] = df['stellar_gravity'] / (df['stellar_radius'] ** 2 + 1e-10)
    result['log_period'] = np.log10(df['period'] + 1)
    
    # Habitable zone
    stellar_luminosity = (df['stellar_radius'] ** 2) * ((df['stellar_temp'] / 5778) ** 4)
    hz_inner = np.sqrt(stellar_luminosity / 1.1)
    hz_outer = np.sqrt(stellar_luminosity / 0.53)
    result['in_habitable_zone'] = ((result['semi_major_axis'] > hz_inner) & 
                                    (result['semi_major_axis'] < hz_outer)).astype(int)
    
    # Planet type
    def classify_planet_type(radius):
        if radius < 1.25:
            return 1  # Earth-like
        elif radius < 2.0:
            return 2  # Super-Earth
        elif radius < 6.0:
            return 3  # Neptune-like
        else:
            return 4  # Jupiter-like
    
    result['planet_type'] = df['planet_radius'].apply(classify_planet_type)
    
    # Interaction feature
    result['period_depth_interaction'] = df['period'] * df['depth']
    
    # Keep disposition column if it was in the original data
    # This allows for retraining in the web app
    
    return result


def load_and_prepare_data(kepler_path, tess_path=None):
    """
    Complete pipeline: Load, standardize, and combine Kepler + TESS data
    
    Returns: Combined DataFrame with standardized column names
    """
    datasets = []
    
    # Load Kepler data
    print("\n📂 LOADING KEPLER DATA...")
    kepler_df, kepler_skip = smart_csv_reader(kepler_path)
    print(f"   Skipped {kepler_skip} comment lines")
    print(f"   Loaded {len(kepler_df)} Kepler stars")
    
    kepler_std, kepler_missing = standardize_columns(kepler_df, 'kepler')
    if kepler_std is not None:
        kepler_std['mission'] = 'Kepler'
        datasets.append(kepler_std)
        print(f"   ✓ Standardized {len(kepler_std.columns)} columns")
    
    # Load TESS data (if provided)
    if tess_path:
        print("\n📂 LOADING TESS DATA...")
        try:
            tess_df, tess_skip = smart_csv_reader(tess_path)
            print(f"   Skipped {tess_skip} comment lines")
            print(f"   Loaded {len(tess_df)} TESS stars")
            
            tess_std, tess_missing = standardize_columns(tess_df, 'tess')
            if tess_std is not None:
                tess_std['mission'] = 'TESS'
                datasets.append(tess_std)
                print(f"   ✓ Standardized {len(tess_std.columns)} columns")
        except FileNotFoundError:
            print(f"   ⚠️  TESS file not found at {tess_path}")
            print(f"   Continuing with Kepler data only...")
    
    # Combine datasets
    if len(datasets) > 1:
        print("\n🔄 COMBINING DATASETS...")
        combined = pd.concat(datasets, ignore_index=True)
        print(f"   Combined: {len(combined)} total stars")
        print(f"   - Kepler: {len(datasets[0])} stars")
        print(f"   - TESS: {len(datasets[1])} stars")
    else:
        combined = datasets[0]
        print("\n   Using Kepler data only")
    
    return combined


def prepare_for_training(combined_df):
    """
    Final preparation: Filter, clean, and engineer features
    
    Returns: X (features), y (labels), feature_names
    """
    # Filter to confirmed labels only
    if 'disposition' in combined_df.columns:
        valid_dispositions = ['CONFIRMED', 'FALSE POSITIVE', 'CANDIDATE', 'NOT DISPOSITIONED']
        combined_df = combined_df[combined_df['disposition'].notna()]
        
        # For Kepler: CONFIRMED and FALSE POSITIVE
        # For TESS: Might use different terms - adapt as needed
        confirmed_terms = ['CONFIRMED', 'CP']
        false_pos_terms = ['FALSE POSITIVE', 'FP']
        
        combined_df = combined_df[
            combined_df['disposition'].str.contains('|'.join(confirmed_terms + false_pos_terms), 
                                                   case=False, na=False)
        ]
    
    # Drop rows with missing base features
    base_features = ['period', 'depth', 'duration', 'planet_radius', 'equilibrium_temp',
                    'insolation_flux', 'stellar_temp', 'stellar_gravity', 'stellar_radius']
    
    combined_df = combined_df.dropna(subset=base_features)
    
    # Create target variable
    combined_df['target'] = combined_df['disposition'].str.contains('CONFIRMED|CP', case=False, na=False).astype(int)
    
    # Engineer all 22 features
    engineered_df = engineer_features(combined_df)
    
    # Select final 22 features for model
    feature_names = [
        'period', 'depth', 'duration', 'planet_radius', 'equilibrium_temp',
        'insolation_flux', 'stellar_temp', 'stellar_gravity', 'stellar_radius',
        'depth_per_duration', 'planet_star_ratio', 'temp_insol_ratio',
        'period_snr', 'depth_snr', 'depth_consistency', 'orbital_speed',
        'semi_major_axis', 'stellar_density', 'log_period', 'in_habitable_zone',
        'planet_type', 'period_depth_interaction'
    ]
    
    # Drop any remaining NaN values
    engineered_df = engineered_df.dropna(subset=feature_names)
    
    X = engineered_df[feature_names]
    y = engineered_df['target']
    mission_info = engineered_df['mission'] if 'mission' in engineered_df.columns else None
    
    return X, y, feature_names, mission_info
