import sys, os
sys.path.append(os.path.abspath('.'))
from src.evaluate import run_evaluation
if __name__ == '__main__':
    print('Executing Lab 7: Modular Pipeline Evaluation...')
    run_evaluation()
