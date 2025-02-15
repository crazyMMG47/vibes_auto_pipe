#!/bin/bash

# Define directories
INPUT_DIR="/home/smooi/vibes_auto_pipe/data/just_bet_nifti"
OUTPUT_DIR="/home/smooi/vibes_auto_pipe/result"
PYTHON_SCRIPT="/home/smooi/vibes_auto_pipe/pipe_model2.py"

# Activate the Conda enviornment 
conda activate tf_env 

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# Iterate over all .nii files in the input directory
for nii_file in "$INPUT_DIR"/*.nii; do
    # Extract the filename without extension
    base_name=$(basename "$nii_file" .nii)

    # Define output file path
    output_nii="$OUTPUT_DIR/${base_name}_mask.nii"

    # Run the Python script with the current .nii file as input
    python3 "$PYTHON_SCRIPT" "$nii_file" "$output_nii"

    echo "Processed: $nii_file -> $output_nii"
done
conda deactivate 

echo "Batch processing completed!"
