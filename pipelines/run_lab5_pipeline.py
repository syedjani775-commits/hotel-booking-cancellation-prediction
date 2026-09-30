import sys, os
sys.path.append(os.path.abspath('.'))
from src.preprocess import run_preprocessing
from pipelines.run_lab3_baseline import train_baselines
if __name__ == '__main__':
    print('Executing Lab 5: Reproducible End-to-End Pipeline...')
    run_preprocessing()
    train_baselines()
