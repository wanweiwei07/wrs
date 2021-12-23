# Functions for promote windows system privilege to Administrator and priority to Realtime

import ctypes, sys

def check_winsys():
    try:
        sys.getwindowsversion()
    except AttributeError:
        return False
    return True

def is_admin():
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False


def promote_admin():
    if is_admin():
        pass
    else:
        # Re-run the program with admin rights
        ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, " ".join(sys.argv), None, 1)

def set_realtime():
    if check_winsys():
        import win32api, win32process
        promote_admin()
        # pid = win32api.GetCurrentProcessId()
        # handle = win32api.OpenProcess(win32con.PROCESS_ALL_ACCESS, True, pid)
        # win32process.SetPriorityClass(handle, win32process.REALTIME_PRIORITY_CLASS)
        win32process.SetPriorityClass(win32api.GetCurrentProcess(), win32process.REALTIME_PRIORITY_CLASS)
    else:
        import os
        pid = os.getpid()
        bash_command = f"chrt -f -p 99 {pid}"
        os.system(bash_command)

if __name__ == '__main__':
    import time
    import numpy as np

    # set_realtime()
    time_list = []
    for i in range(1000):
        tic=time.time()
        for i in range(1000):
            a=1+2
        # time.sleep(.001)
        toc=time.time()
        time_list.append(toc-tic)
    print(np.mean(time_list), np.std(time_list))