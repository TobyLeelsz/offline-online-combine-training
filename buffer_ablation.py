import subprocess

def buffer_n(n:int):
    '''
    n: int - number of episodes to generate offline dataset
    '''

    # List of scripts to run in order

    scripts = [
        ["SAC-Online\\generate_dataset.py", "--n_episode", f"{n}"],
        ["CQL-SAC-Combine\\train.py", "--episodes", "200", "--n_episode", f"{n}"],
        ["CQL-SAC-Combine\\eval.py", "--episodes", "200",  "--n_episode", "100"]
    ]

    # Run each script
    for script in scripts:
        try:
            subprocess.run(["python"] + script, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error occurred while running {script[0]}: {e}")
            exit(1)

if __name__ == '__main__':
    for n in range(50,711,110):
        buffer_n(n)