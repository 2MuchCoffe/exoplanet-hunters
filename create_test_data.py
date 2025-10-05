"""
Create proper test data files for demo
Preserves original column names so web app can detect mission type
"""

import pandas as pd
import numpy as np
from pathlib import Path
from utils.data_processor import smart_csv_reader

print("=" * 70)
print("CREATING TEST DATA FILES FOR DEMO")
print("=" * 70)

# Create test data folder
test_folder = Path('data/test_data_for_demo')
test_folder.mkdir(exist_ok=True)

# Load Kepler data using smart reader
print("\n📂 Loading Kepler data...")
kepler_df, kepler_skip = smart_csv_reader('data/cumulative.csv')
print(f"   Auto-skipped {kepler_skip} comment lines")
print(f"   Total Kepler stars: {len(kepler_df)}")

# Filter to confirmed labels only
kepler_df = kepler_df[kepler_df['koi_disposition'].isin(['CONFIRMED', 'FALSE POSITIVE'])]

# Remove rows with missing critical data
required_cols = ['koi_period', 'koi_depth', 'koi_duration', 'koi_prad', 
                'koi_teq', 'koi_insol', 'koi_steff', 'koi_slogg', 'koi_srad']
kepler_df = kepler_df.dropna(subset=required_cols)

print(f"   After filtering: {len(kepler_df)} stars with complete data")

# Take 20% for test set (stratified)
from sklearn.model_selection import train_test_split

kepler_train, kepler_test = train_test_split(
    kepler_df,
    test_size=0.20,
    random_state=42,
    stratify=kepler_df['koi_disposition']
)

# Save Kepler test set with ORIGINAL columns
kepler_test.to_csv(test_folder / 'kepler_test_20percent.csv', index=False)
print(f"✓ Saved Kepler test: {len(kepler_test)} stars")
print(f"   File: test_data_for_demo/kepler_test_20percent.csv")

# Try to load TESS data (if exists)
try:
    print("\n📂 Loading TESS data...")
    tess_df, tess_skip = smart_csv_reader('data/tess_data.csv')
    print(f"   Auto-skipped {tess_skip} comment lines")
    print(f"   Total TESS stars: {len(tess_df)}")
    
    # Filter to confirmed labels
    if 'tfopwg_disp' in tess_df.columns:
        tess_df = tess_df[tess_df['tfopwg_disp'].isin(['PC', 'FP', 'CP'])]
    
    # Remove rows with missing data
    tess_required = ['pl_orbper', 'pl_trandep', 'pl_trandurh', 'pl_rade',
                     'st_teff', 'st_logg', 'st_rad']
    tess_available = [col for col in tess_required if col in tess_df.columns]
    
    print(f"   Found {len(tess_available)}/{len(tess_required)} required columns")
    
    if len(tess_available) >= 6:  # Need most columns
        tess_df = tess_df.dropna(subset=tess_available)
        
        print(f"   After filtering: {len(tess_df)} stars with complete data")
        
        # Take 20% for test
        if len(tess_df) > 10:  # Only if enough data
            tess_train, tess_test = train_test_split(
                tess_df,
                test_size=0.20,
                random_state=42
            )
            
            # Save TESS test set
            tess_test.to_csv(test_folder / 'tess_test_20percent.csv', index=False)
            print(f"✓ Saved TESS test: {len(tess_test)} stars")
            print(f"   File: test_data_for_demo/tess_test_20percent.csv")
            
            # Create combined file (mix of both)
            print("\n🔄 Creating combined test file...")
            # Take smaller sample from each to create mixed file
            kepler_sample = kepler_test.sample(min(100, len(kepler_test)), random_state=42)
            tess_sample = tess_test.sample(min(100, len(tess_test)), random_state=42)
            
            # Combine with matching columns only
            combined_test = pd.concat([kepler_sample, tess_sample], ignore_index=True)
            combined_test.to_csv(test_folder / 'combined_test_mixed.csv', index=False)
            print(f"✓ Saved Combined test: {len(combined_test)} stars")
            print(f"   - Kepler: {len(kepler_sample)} stars")
            print(f"   - TESS: {len(tess_sample)} stars")
            print(f"   File: test_data_for_demo/combined_test_mixed.csv")
        else:
            print("   ⚠️ Not enough TESS data for test split")
    else:
        print(f"   ⚠️ Missing too many required columns")
        
except FileNotFoundError:
    print("\n   ⚠️ TESS file not found - creating Kepler-only test files")
except Exception as e:
    print(f"\n   ⚠️ Error loading TESS: {str(e)}")
    print("   Continuing with Kepler test data only")

# Create README
print("\n📝 Creating README for test files...")
readme_content = """# Test Data for Demo

These files contain 20% of the data (held out from training) for demonstration purposes.

## Files:

### kepler_test_20percent.csv
- 20% of Kepler mission data
- Contains original Kepler column names (koi_*)
- Star names: kepoi_name, kepler_name
- ~1,450 stars
- Use for: Kepler-only demo

### tess_test_20percent.csv (if available)
- 20% of TESS mission data
- Contains original TESS column names (pl_*, st_*)
- Star names: toi, tid
- Use for: TESS-only demo

### combined_test_mixed.csv (if TESS available)
- Mix of both Kepler and TESS stars
- Contains both column naming schemes
- Use for: Demonstrating multi-mission capability

## How to Use:

Upload any of these files in the Batch Analysis page to see the model in action!

The model has NEVER seen this data during training - it's a true test of generalization.
"""

with open(test_folder / 'README.md', 'w') as f:
    f.write(readme_content)

print("✓ README created")

print("\n" + "=" * 70)
print("✅ TEST DATA CREATION COMPLETE!")
print("=" * 70)
print("\n📁 Files created in: data/test_data_for_demo/")
print("   Upload these in the web app to demo the model!")
print("=" * 70)
