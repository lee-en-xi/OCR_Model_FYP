import random
import os

# Configuration
train_file = 'train.txt'
val_file = 'val.txt'
output_dir = '.'
new_train_file = os.path.join(output_dir, 'new_train.txt')
new_val_file = os.path.join(output_dir, 'new_val.txt')
val_ratio = 0.2  # 20% for validation
random_seed = 42  # For reproducibility

# Read and combine all data
all_samples = []

# Read train file
with open(train_file, 'r') as f:
    train_samples = f.readlines()
    all_samples.extend(train_samples)

# Read validation file
with open(val_file, 'r') as f:
    val_samples = f.readlines()
    all_samples.extend(val_samples)

# Remove empty lines and strip whitespace
all_samples = [line.strip() for line in all_samples if line.strip()]

# Shuffle the data
random.seed(random_seed)
random.shuffle(all_samples)

# Split into new train/val sets
split_idx = int(len(all_samples) * (1 - val_ratio))
new_train_samples = all_samples[:split_idx]
new_val_samples = all_samples[split_idx:]

# Write new train file
with open(new_train_file, 'w') as f:
    f.write('\n'.join(new_train_samples))

# Write new validation file
with open(new_val_file, 'w') as f:
    f.write('\n'.join(new_val_samples))

print(f"Remixing complete!")
print(f"Original samples: Train={len(train_samples)}, Val={len(val_samples)}")
print(f"New samples: Train={len(new_train_samples)}, Val={len(new_val_samples)}")
print(f"New files saved to: {new_train_file} and {new_val_file}")