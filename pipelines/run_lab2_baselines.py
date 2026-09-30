import sys, os
sys.path.append(os.path.abspath('.'))
from pipelines.run_lab3_baseline import train_baselines
if __name__ == '__main__':
    print('Executing Lab 2: Baseline Model Training...')
    train_baselines()
