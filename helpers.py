import os
import numpy as np
import pandas as pd
import glob
from sklearn.preprocessing import RobustScaler


# --------------------
# Data Loading Helpers
# --------------------

# Load and Label Helpers
def benign_load_and_label(all_csvs):
    '''
    Finds all csv's with "Benign" at the beginning of the filename, concatenates, and labels them

    '''
    # Benign files
    benign_files_flow = [f for f in all_csvs if os.path.basename(f).startswith("Benign")]

    # concatenates benign data, and label creation
    df_benign_flow = pd.concat(
        (pd.read_csv(f).assign(label='benign') for f in benign_files_flow), 
        ignore_index=True
    )
    return df_benign_flow.copy()

def attacks_load_and_label(all_csvs):
    """
    Finds, loads, and labels all attack files from a list of CSV paths.
    """
    # Attack files
    attack_files = [f for f in all_csvs if not os.path.basename(f).startswith("Benign")]
    substring_map = {
        'DDoS-HTTP': 'DDOS-HTTP_flood',
        'DoS-HTTP': 'DoS-HTTP',
        'Spoofing': 'DNS_spoofing',
        'XSS': 'XSS',
        'BruteForce': 'brute_force'
    }
    
    processed_dfs = []
    for file_path in attack_files: # Loop through files, load, label, and append to the list
        df = pd.read_csv(file_path, low_memory=False)
        filename = os.path.basename(file_path)
        
        # label based on file name
        assigned_label = 'Unknown Attack'
        for substring, label in substring_map.items():
            if substring in filename:
                assigned_label = label
                break       
        # Apply the label and store the dataframe
        df['label'] = assigned_label
        processed_dfs.append(df)
        
    # Combine the list into a dataframe
    return pd.concat(processed_dfs, ignore_index=True).copy()      

def benign_sampler_200k(df_benign, seed=0):
    '''
    Randomly samples 200, 000 benign packets
    '''
    if len(df_benign) < 200_000: # integrity check
        raise ValueError(f"Insufficient benign data. Expected >200,000 rows, found {len(df_benign)}.")
        
    df_benign_sample = df_benign.sample(n=200_000, random_state=seed)
    return df_benign_sample



def attack_sampler(df_attacks, label_col='label', random_seed=0):
    """
    4000-6200 attack observations sampled, plus integrity checks.
    Sampling is stratified to ensure equal representation of each attack
    """
    # Integrity Check
    unique_attack_count = df_attacks[label_col].nunique()
    if unique_attack_count != 5:
        raise ValueError(f"Expected 5 attack types, found {unique_attack_count}.")

    # generate random sample
    rng = np.random.default_rng(seed=random_seed)
    num_attack_samples = rng.integers(4000, 6200, endpoint=True)
    samples_per_attack = num_attack_samples // 5 # close enough
    # Group by and sample
    df_attack_sampled = (df_attacks.groupby(label_col)
                                    .sample(n=samples_per_attack, random_state=random_seed))
    
    # Integrity check
    total_sampled = len(df_attack_sampled)
    # Within the required bounds?
    assert 4000 <= total_sampled <= 6200, f"Sample size {total_sampled} out of bounds [4000-6200]."
    # 5 classes represented equally?
    assert df_attack_sampled[label_col].nunique() == 5, "Output is missing attack classes."
    # missing labels?
    assert df_attack_sampled[label_col].isna().sum() == 0, "Null values found in label column."
    
    return df_attack_sampled 

def sample_traffic(df_benign, df_attack):
    '''
    samples as requested, and combines results
    '''
    df_sampled_benign_packet = benign_sampler_200k(df_benign) # benign flow
    df_sampled_attack_packet = attack_sampler(df_attack) # attack flow

    # Combine the sampled datasets
    df_combined_sampled = pd.concat([df_sampled_benign_packet, df_sampled_attack_packet], ignore_index=True)
    return df_combined_sampled

def load_csv(path):
    '''
    Finds all .csv files in given path.
    Combines benign files and samples 200 000 rows as requested.
    Combines all attack files and sample 4000-6200 as requested.
    Combines and returns benign and attack records
    '''
    all_csvs = glob.glob(os.path.join(path, "*.csv"))
    # Begign files
    df_benign = benign_load_and_label(all_csvs) # flow based benign data
    # Attack files
    df_attack = attacks_load_and_label(all_csvs) # flow based attack data
    df_combined_sampled = sample_traffic(df_benign, df_attack)
    return df_combined_sampled  

# --------------------------------------------
# Validation Helpers, not used in final script
# --------------------------------------------

def check_for_infinities(df):
    """
    What it says on the tin
    """
    # numeric columns, others can't be infinity
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Check for positive and negative infinity
    inf_counts = df[numeric_cols].isin([np.inf, -np.inf]).sum()
    
    # columns that contain infinities
    inf_cols = inf_counts[inf_counts > 0].sort_values(ascending=False)
    
    print("--- Infinite Values Report ---")
    if inf_cols.empty:
        print("No infinite values found in the dataset.")
    else:
        print(f"Found infinities in {len(inf_cols)} columns:\n")
        print(inf_cols.to_string())
        
    return inf_cols

def identify_outliers(df_combined, multiplier=1.5):
    """
    What it says on the tin
    """
    # Numeric columns 
    df_numeric = df_combined.select_dtypes(include=[np.number])
    
    # First and Third Quantile or 25th and 75th percentile
    Q1 = df_numeric.quantile(0.25)
    Q3 = df_numeric.quantile(0.75)
    IQR = Q3 - Q1 # Inter Quartile Range
    
    # Statistical boundaries for "Outlier"
    lower_bound = Q1 - (multiplier * IQR)
    upper_bound = Q3 + (multiplier * IQR)
    
    # Mask of values outside the boundaries
    outlier_mask = (df_numeric < lower_bound) | (df_numeric > upper_bound)
    
    # Count outliers
    outlier_counts = outlier_mask.sum()
    outlier_pct = (outlier_counts / len(df_numeric)) * 100
    
    # Human readable
    report = pd.DataFrame({
        'Outlier Count': outlier_counts,
        'Percentage (%)': outlier_pct
    })
    
    # Features with outliers, descending order
    report = report[report['Outlier Count'] > 0].sort_values(by='Percentage (%)', ascending=False)
    return report


def get_missing_report(df, dataset_name):
    """
    Finds and reports missing values
    """
    # Calculate the percentage of NaNs per column
    missing_pct = df.isna().mean() * 100
    
    # Filter for columns that have at least one missing value and sort
    missing_cols = missing_pct[missing_pct > 0].sort_values(ascending=False)
    
    print(f"--- {dataset_name} ---")
    if missing_cols.empty:
        print("No missing data found.\n")
    else:
        print(missing_cols.to_string())
        print("\n")

# ---------------------
# Preprocessing Helpers
# ---------------------

def shuffle_and_seperate(df_combined, seed =0):
    """
    shuffles features, and splits features from label column
    """
    # Shuffle
    df_combined = df_combined.sample(frac=1, random_state=seed).reset_index(drop=True)
    # Labels
    labels = df_combined['label']
    # Data
    df_features = df_combined.drop(columns=['label'])
 
    return df_features, labels


def handle_missing_data(df_features):
    '''
    Missing data feature engineering

    * Missing data like containsuseful information
    * Each column with missing data has a new feture column created,
    * It is a binary variable with a value of 1 if there was a missing value
    * median replaces Nan if numeric column
    * new category is created if discrete column, no new feature column is needed.
    * The big idea is to impute values for missing data so models can train, and also keep track of if the data was imputed
    '''
    df_clean = df_features.copy()
    
    # map placeholders "-1", and -1 to NaN
    df_clean = df_clean.replace({"-1": np.nan, -1: np.nan})
    # splits columns by numeric vs discrete
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    categorical_cols = df_clean.select_dtypes(exclude=[np.number]).columns
    
    #  Infinity handling
    # Columns that contain infinity
    inf_mask = df_clean[numeric_cols].isin([np.inf, -np.inf])
    inf_cols = inf_mask.any()[inf_mask.any()].index # columns with infinity
    for col in inf_cols: # for columns with an infinity
        # create indicator features to preserve information of 'Infinity'
        df_clean[f"{col}_is_infinite"] = df_clean[col].isin([np.inf, -np.inf]).astype(int)
        # maximum observed value in this column
        max_finite_val = df_clean.loc[df_clean[col] != np.inf, col].max()
        # Cap infinities at maximum observed value. this is imputation for infinity
        df_clean[col] = df_clean[col].replace(np.inf, max_finite_val)
    
    # NaN Handling
    # find columns with missing values
    missing_numeric_pct = df_clean[numeric_cols].isna().mean()
    numeric_missing_cols = missing_numeric_pct[missing_numeric_pct > 0].index
    for col in numeric_missing_cols: # for columns with missing values
        df_clean[f"{col}_is_missing"] = df_clean[col].isna().astype(int) # creates dummy featues 
        
    # Imputes median for numeric columns, and unknown for categorical
    df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())
    df_clean[categorical_cols] = df_clean[categorical_cols].fillna('unknown')
    
    return df_clean

def one_hot_encoder(df):
    """
    one hot encodes categorical features with low cardinality, these were discovered during EDA
    """
    df = df.drop(columns=['http_host'], errors='ignore')  # this column was found to only contain 'none', and has no information
    cols_to_encode = ['http_request_method', 'handshake_version', 'http_content_type']
    
    df_encoded = pd.get_dummies( 
        df, 
        columns=[c for c in cols_to_encode if c in df.columns], 
        dtype=int 
    )
    return df_encoded

def split_identifiers(df_clean):
    '''
    Finds columns that identify a machine or user to seperate them from training.
    They are not a measure of network behaviour, should not be used in training.
    '''
    # list of magic words to pick out columns that identify a machine or user. Discovered by inspection
    identifier_keywords = [
        'ip', 'port', 'mac', 'timestamp', 'flow_id', 'protocol', 
        'server', 'host', 'user_agent', 'oui', 'uri', 'content_type'
    ]
    identifier_cols = []
    for col in df_clean.columns:
        # Break the column name into words based on delimiters
        words = col.lower().replace(' ', '_').replace('-', '_').split('_')
        # Check if the keyword is a word in the column name
        if any(keyword in words for keyword in identifier_keywords):
            identifier_cols.append(col)
        
    df_identifiers = df_clean[identifier_cols].copy()
    df_features_clean = df_clean.drop(columns=identifier_cols).copy()
    
    return df_features_clean, df_identifiers

def feature_cleaner(df_features):
    """
    handles missing data, one-hot encodes low cardinality categorical features, and selects features to train on
    """
    df_clean_features = handle_missing_data(df_features) # no need to worry about labels
    df_clean_hot = one_hot_encoder(df_clean_features)
    df_features_clean, df_identifiers = split_identifiers(df_clean_hot) # For clarity
    return df_features_clean, df_identifiers


def log_and_scale(df_features_clean, labels):
    """
    Apprlies a natural logarithm to numeric features, to help with skewness seen in fetaure histograms.
    Scales features non-paramtericly, this is to avoid bias.
    Returns the pre-processed data, along with scaler needed for testing.
    """
    #  Drop columns that are un-parsable text
    df_features_clean = df_features_clean.dropna(axis=1, how='all')
    df_features_clean = df_features_clean.select_dtypes(include=[np.number])
    
    # Log Transformation
    # Compresses heavy-tailed distributions 
    df_log_transformed = np.log1p(df_features_clean)
    
    # Robust Scaling
    # Uses median and interquartile range to scale data, this is non-parmetric.
    scaler = RobustScaler()
    scaled_array = scaler.fit_transform(df_log_transformed)
    df_scaled = pd.DataFrame(scaled_array, columns=df_features_clean.columns)
    
    # Reassemble the dataset
    df_final = pd.concat([
        df_scaled.reset_index(drop=True), 
        labels.reset_index(drop=True)
    ], axis=1)
    
    return df_final, scaler