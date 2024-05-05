import subprocess

# List of scripts to run in order
scripts = [
    "SAC-Online\\train.py",
    "SAC-Online\\generate_dataset.py",
    "CQL-SAC-Combine\\train.py",
    "CQL-SAC-Combine\\eval.py"
]

# Run each script
for script in scripts:
    subprocess.run(["python", script])