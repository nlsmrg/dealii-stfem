import json
from argparse import ArgumentParser
import sys
from hashlib import blake2b # just a fast hash function

def generate_hash(args):
    # Create a unique string from all arguments
    arg_str = f"{args.executable}{args.job}{args.nodes}{args.tpern}{args.part}{args.cpupt}{args.lim}{args.param}{args.dim}"
    if args.precon_float:
        arg_str += "precondition_float"
    hash_object = blake2b(digest_size=3)
    hash_object.update(arg_str.encode())
    return hash_object.hexdigest()

def generate_slurm_script(args):
    unique_id = generate_hash(args)
    base_filename = f"job_{unique_id}"
    script_content = f"""#!/bin/bash
#SBATCH --job-name={args.job} # specifies a user-defined job name
#SBATCH --nodes={args.nodes} # number of compute nodes to be used
#SBATCH --ntasks-per-node={args.tpern} # number of MPI processes
#SBATCH --partition={args.part} # partition (small, medium, small_fat, small_gpu)
#SBATCH --cpus-per-task={args.cpupt} # number of cores per process
#SBATCH --time={args.lim} # maximum wall clock limit for job execution
#SBATCH --output={base_filename}.log # log file which will contain all output

# commands to be executed
srun  --mpi=pmix {args.executable} --file {args.param} --dim {args.dim} --log_prefix {base_filename}"""
    if args.precon_float:
        script_content += " --precondition_float 1"

    script_content += "\n"
    with open(args.param, 'r') as f:
        parameters = json.load(f)
    for key, value in vars(args).items():
        parameters[key] = value
    with open(f"{base_filename}_params.json", 'w') as f:
        json.dump(parameters, f, indent=4, separators=(',', ': '))

    with open(f"{base_filename}.sh", 'w') as f:
        f.write(script_content)

    print(f"{base_filename}.sh")

def main():
    parser = ArgumentParser(description='Generate a Slurm job script.')
    parser.add_argument('--job', type=str, default='stmg', help='Job name')
    parser.add_argument('--nodes', type=int, required=True, help='Number of nodes')
    parser.add_argument('--tpern', type=int, default=72, help='Tasks per node')
    parser.add_argument('--part', type=str, required=True, help='Partition name (small, medium, small_fat, small_gpu)')
    parser.add_argument('--cpupt', type=int, default=1, help='CPUs per task')
    parser.add_argument('--lim', type=str, default='24:00:00', help='Time limit')
    parser.add_argument('--param', type=str, default='default', help='Path to the parameter file')
    parser.add_argument('--dim', type=int, default=3, help='Dimension of the problem')
    parser.add_argument("--precon_float", action='store_true', help="Perform precondition in float precision")
    parser.add_argument('--executable', type=str, default='./tests/tp_01.release/tp_01.release', help='Path to the executable')
    args = parser.parse_args()
    generate_slurm_script(args)

if __name__ == '__main__':
    main()
