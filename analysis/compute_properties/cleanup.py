"""Clean temporary working folders created to `data/_tmp_*/`"""
import os
import sys
import shutil
from pathlib import Path


# make sure script can be run from anywhere
SCRIPT_PATH = Path(__file__).parent.resolve()
DATA_FOLDER = SCRIPT_PATH.parent.parent / 'data'


def cleanup():
    """Clean temporary working folders created to `data/_tmp_*/`"""
    # get all temporary working directories
    tmp_dirs = [d for d in DATA_FOLDER.iterdir() if d.is_dir() and d.name.startswith('_tmp_')]
    if len(tmp_dirs) == 0:
        print('No temporary working directories found.')
        sys.exit(0)
    
    # ask for confirmation
    print(f'Found {len(tmp_dirs)} temporary working directories:')
    for d in tmp_dirs:
        print(f'  {d}')
    print('Do you want to delete them? (y/n)')
    answer = input()
    if answer.lower() != 'y':
        print('Aborting.')
        sys.exit(0)
    
    # delete temporary working directories
    for d in tmp_dirs:
        print(f'Deleting {d}...')
        shutil.rmtree(d)
    print('Done.')

    attribution_log_files = []
    if (DATA_FOLDER / 'attributions').exists():
        attribution_log_files = [
            f for f in (DATA_FOLDER / 'attributions').iterdir() 
            if f.name.startswith('log_') and f.name.endswith('.txt')
        ]
    if len(attribution_log_files) > 0:
        print(f'Found {len(attribution_log_files)} attribution log files:')
        for f in attribution_log_files:
            print(f'  {f}')
        print('Do you want to delete them? (y/n)')
        answer = input()
        if answer.lower() != 'y':
            print('Aborting.')
            sys.exit(0)
        for f in attribution_log_files:
            print(f'Deleting {f}...')
            os.remove(f)
        print('Done.')

    sys.exit(0)


if __name__ == '__main__':
    import fire
    fire.Fire(cleanup)
