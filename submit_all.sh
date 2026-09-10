#!/bin/bash

densities=(0.00 0.10 0.50 1 10 15 25 50 75 100 250 500 750 999.99)

mkdir -p logs

for d in "${densities[@]}"; do
    sbatch <<EOF    
#!/bin/bash
#SBATCH --job-name=dcll-natenr-U238-${d}
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=4G
#SBATCH --time=24:00:00
#SBATCH --mail-type=start,fail
#SBATCH --mail-user=gl3210@princeton.edu

module purge
module load anaconda3/2025.6
conda activate buclear-nomb
cd dt-fusion-illicit/
python -u main.py -r tallies -b DCLL -i U238 -p 4e6 -c 25 -f ${d} -l 7.5
EOF
done