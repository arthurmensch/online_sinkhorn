#!/bin/env bash
#SBATCH --job-name=toy
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --mem-per-cpu=4G
#SBATCH --hint=nomultithread
#SBATCH --time=2:00:00
#SBATCH --output=run_%j.out
#SBATCH --error=run_%j.out
#SBATCH --signal=B:SIGUSR1@120
#SBATCH -A qxo@cpu

sig_handler_USR1()
{
        echo "Will die in 120s"  #. Rescheduling"
#        sbatch $0
        exit 2
}

trap 'sig_handler_USR1' SIGUSR1

source load_modules

python benchmark_finite.py &

wait

exit 0

