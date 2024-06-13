'''
Helper function for llamafile
'''
# import
import os
import wget
import time
import subprocess
from signal import SIGKILL
from utilities import getconfig
# definition default
CONFIG = "emb-llamafile"
PATH_DATA = getconfig(CONFIG)["pathdata"]
PATH_PID = os.path.join(PATH_DATA, ".llamafile_pid")
LLAMAFILE = getconfig(CONFIG)["embedmodel"]
PATH_MDL  = os.path.join(PATH_DATA, LLAMAFILE)
URL_MDL = getconfig(CONFIG)["embedmodel_url"]
def launch_llamafile(path_mdl=PATH_MDL, url_mdl=URL_MDL, time_sleep=5, config=None):
    '''
    Launch process but do not wait for it to finish
    '''
    # clean
    kill_llamafile()
    # check if config is provided
    if config is not None:
        path_mdl, url_mdl = get_llamafile_conf(config)
    # check model file or download
    if not os.path.isfile(path_mdl):
        path_dl = wget.download(url_mdl, out=PATH_DATA)
        print(f"Downloaded {path_dl}")
        os.rename(path_dl, path_mdl)
        # change chmod to be executable
        os.chmod(path_mdl, os.stat(path_mdl).st_mode | 0o111)
        
    # prepare command
    str_cmd = f'{path_mdl} --server --nobrowser --embedding'
    print(str_cmd)
    # run command
    process = subprocess.Popen(
        str_cmd,
        shell=True,
    )
    # write llamafile server PID into a file
    with open(PATH_PID, 'w', encoding='utf-8') as f:
        f.write(str(process.pid))
    # wait for the process to be "ready"
    time.sleep(time_sleep)

    return process.pid


def kill_llamafile():
    '''
    Kill process
    # cleanup: kill the llamafile server process
    '''
    if os.path.isfile(PATH_PID):
        # kill process
        with open(PATH_PID, 'r', encoding='utf-8') as f:
            pid = int(f.read().strip())
        print(f"Killing PID {pid}...")
        try:
            os.kill(pid, SIGKILL)
        except ProcessLookupError:
            print(f"Process {pid} not found")
        # remove file
        try:
            os.remove(PATH_PID)
        except FileNotFoundError:
            pass
def get_llamafile_conf(config=CONFIG):
    '''
    Get llamfile config
    '''
    pathdata = getconfig(config)["pathdata"]
    embedmodel = getconfig(config)["embedmodel"]
    embedmodel_url = getconfig(config)["embedmodel_url"]
    return os.path.join(pathdata, embedmodel), embedmodel_url

