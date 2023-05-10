import datetime
import os

def make_run_name(CFG):
    tz = datetime.timezone(datetime.timedelta(hours=9))
    day_time = datetime.datetime.now(tz=tz)
    run_name = day_time.strftime(f'%m%d_%H:%M:%S_{CFG.admin}')
    dir_path = f'./results/{run_name}'
    dir_path_log = f'./results/{run_name}/log'

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    if not os.path.exists(dir_path_log):
        os.makedirs(dir_path_log)
    
    return dir_path
    