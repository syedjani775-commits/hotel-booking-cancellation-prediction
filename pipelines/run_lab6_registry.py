import sys, os
sys.path.append(os.path.abspath('.'))
from pipelines.run_lab4_tracking import track_experiments
if __name__ == '__main__':
    print('Executing Lab 6: Model Registry...')
    track_experiments()
