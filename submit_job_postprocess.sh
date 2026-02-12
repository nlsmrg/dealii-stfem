#!/bin/bash
declare -a output_dirs=()
declare -a script_files=()
# Function to submit jobs
submit_job() {
    local param_file=$1
    local clean_name=$(basename "${param_file%_*}")
    local job_name="$clean_name"
    local n_nodes=$2
    local partition=$3
    local limit=$4
    local precondition_float=$5
    local executable=$6
    local script_file=$(python job_generator.py --executable $executable --job $job_name --nodes $n_nodes --part $partition --lim $limit --param $param_file --dim 3 $precondition_float)
    script_files+=(script_file)
    # Submit job and capture job ID
    local sbatch_output=$(sbatch "$script_file")
    echo "$sbatch_output"
    local job_id=$(echo "$sbatch_output" | awk '{print $4}')

    # Wait for the job to finish
    while squeue -j "$job_id" | grep -q "$job_id"; do
        sleep 60 # Check every minute if the job is still in the queue
    done
    sleep 30 # to be sure file operations are completed
    local script_base_name=$(basename $script_file .sh)

    # Move output to designated directory
    local output_dir="output/$clean_name/$script_base_name/"
    output_dirs+=(output_dir)
    echo $output_dir
    mkdir -p "$output_dir"
    awk '/^Convergence table/ { print_it = 1 } print_it { if (/^[a-zA-Z]/) print "#" $0; else print $0 } /^$/ { print_it = 0 }' "${script_base_name}.log" > "$output_dir/convergence_tables.txt"
    awk '/^Iteration count table/ { print_it = 1 } print_it { if (/^[a-zA-Z]/) print "#" $0; else print $0 } /^$/ { print_it = 0 }' "${script_base_name}.log" > "$output_dir/iteration_tables.txt"
    mv solution*.pvtu "$output_dir"
    mv solution*.vtu "$output_dir"
    mv "${script_base_name}*.log" "$output_dir"
    mv functionals.txt "$output_dir"
    mv "${script_base_name}_params.json" "$output_dir"
}
