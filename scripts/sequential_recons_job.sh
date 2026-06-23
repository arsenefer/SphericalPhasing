#!/bin/bash

#!/bin/bash
# SLURM options:
# Job name and logs
#SBATCH --job-name=heron_adf_%j
#SBATCH --output=../logs/heron_adf_%j.log

# Partition and resources
#SBATCH --partition=htc
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=20000
#SBATCH --time=0-05:00:00

# Notification
#SBATCH --mail-user=arsene.ferriere@lpnhe.in2p3.fr
#SBATCH --mail-type=END,FAIL

# Licenses (if needed)
#SBATCH --licenses=sps

set -euo pipefail

# Activate environment
source /sps/grand/aferrier/ccenvtest/bin/activate

# If user provided NPROCS env var, use it; else allow passing --nprocs to python

# Forward all arguments to the python script
echo "Running full_recons.py with args: $@"
python scripts/sequential_recons.py "$@"