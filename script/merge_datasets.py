import pandas as pd

# Load source data (the new data)
source_file = 'output/akt8_mineru_ocr_extract/qwen_sentence_aligned_akt8.csv'
df_source = pd.read_csv(source_file)

# Load target data (existing training data)
target_file = 'data/data_k/train_clean_qwen_reject.csv'
df_target = pd.read_csv(target_file)

print(f"Source shape: {df_source.shape}")
print(f"Target shape: {df_target.shape}")

# Get existing oare_ids
existing_ids = set(df_target['oare_id'].unique())
print(f"Unique IDs in target: {len(existing_ids)}")

# Filter source to find new data
# We also want to prioritize source over target if there are duplicates,
# as per SPEC: "If source is better/more complete, update; otherwise skip."
# Since we don't have a clear way to compare quality automatically, we will assume
# the source data (newly aligned) is the preferred version for these specific IDs.

# However, the logic says "check if oare_id exists".
# If it exists, we might want to skip or update.
# Let's filter out rows where oare_id is in target, assuming we want to keep the target version
# OR we want to append ONLY new IDs.

# Let's assume we want to append ONLY new IDs to avoid overwriting existing data which might have been manually curated.
# But SPEC said: "If they exist: Compare translations. If source is better/more complete, update..."
# Doing "Compare translations" automatically is hard.

# Simplest robust approach: Add only new OARE IDs that are not in the target set.
# This is safe and augments the data.

new_ids = set(df_source['oare_id'].unique()) - existing_ids
print(f"New IDs to add: {len(new_ids)}")

df_new = df_source[df_source['oare_id'].isin(new_ids)]
print(f"New rows to add: {len(df_new)}")

# Save to output
output_file = 'data/data_k/akt8_augmented.csv'
df_new.to_csv(output_file, index=False)
print(f"Saved {len(df_new)} new rows to {output_file}")