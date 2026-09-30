import sys, os
sys.path.append(os.path.abspath('.'))
from src.preprocess import run_preprocessing
if __name__ == '__main__':
    print('Executing Lab 1: Dataset Ingestion & Preprocessing...')
    run_preprocessing()
