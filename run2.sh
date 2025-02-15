#!/bin/bash

# Define directories
INPUT_DIR="/home/smooi/vibes_auto_pipe/data/Cc_just_bet_nifti"
OUTPUT_DIR="/home/smooi/vibes_auto_pipe/result"
PYTHON_SCRIPT="/home/smooi/vibes_auto_pipe/pipe_model2.py"

# Ensure Conda is properly initialized
source /home/smooi/miniconda3/etc/profile.d/conda.sh  # Correct Conda path

# Activate the Conda environment
conda activate tf_env || { echo "Failed to activate Conda environment"; exit 1; }

# Print Python path to confirm it's from the correct environment
echo "Using Python: $(which python)"
python -c "import sys; print(sys.path)"

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# Iterate over all .nii files in the input directory
for nii_file in "$INPUT_DIR"/*.nii; do
    # Extract the filename without extension
    base_name=$(basename "$nii_file".nii)

    # Define output file path
    output_nii="$OUTPUT_DIR/${base_name}_mask.nii"

    # Use the full path to Python from the Conda environment # TODO: try with this
    /home/smooi/miniconda3/envs/tf_env/bin/python "$PYTHON_SCRIPT" "$nii_file" "$output_nii"
    
    echo "Processed: $nii_file -> $output_nii"
done

# Deactivate Conda environment after processing
conda deactivate

echo "Batch processing completed!"
