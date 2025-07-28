
from datetime import datetime
import os

SLURM_JOB_ID = os.getenv("SLURM_JOB_ID", "LOCAL")  # “LOCAL” when not on Slurm

def build_run_name_suffix():
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{ts}_J{SLURM_JOB_ID}"