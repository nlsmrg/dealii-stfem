#!/bin/bash
source ./generate_parameters.sh

problems="wave heat"
types="DG CGP"
distortGrids="0.0 0.15"

# Run the function and store the output filenames
generate_convergence_parameters "$problems" "$types" "$distortGrids" "1"

source ./submit_job_postprocess.sh

# Run the compute jobs
for file in "${filenames_c[@]}"; do
    submit_job "$file" 96 medium "24:00:00" "--precon_float"
done
