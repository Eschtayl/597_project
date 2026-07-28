from helpers import *

path = 'packet_based' # Path to packet files. Uncomment below for flow based
#path = 'flow_based' # path to flow flies

# Load benign and attack csvs, and sample as reqested
df_combined_sampled = load_csv(path)

# Shuffles data and splits features from labels, so features can be cleaned
df_features, labels = shuffle_and_seperate(df_combined_sampled)

# Seperates network behaviour features, from features that identify a machine or user. Numeric features cleaned.
df_features_clean, df_identifiers = feature_cleaner(df_features)

# numeric fetaures are passed through a logarithm then scaled. label and identifier columns are recombined into final df_preprocessed result
# fitted_scaler, computed during training, is saved to be used with test data
df_preprocessed, fitted_scaler = log_and_scale(df_features_clean, labels)

print(f"Final preprocessed shape: {df_preprocessed.shape}")
print(f"Total NaNs in final dataset: {df_preprocessed.isna().sum().sum()}") # integrity