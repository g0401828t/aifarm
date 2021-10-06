import pickle
import os
import time
from colorama import Fore, Back, Style

import torch
import psutil

# pip install gpustat


class command():
    def __init__(self, command_str, status, end_time, elapsed_time):
        self.command_str = command_str
        self.status = status
        self.end_time = end_time
        self.elapsed_time = elapsed_time


torch_version = torch.__version__
torch_cuda_version = torch.version.cuda
torch_cuda_available = torch.cuda.is_available()

while True:

    with open("log.pkl", "rb") as f:
        c_list = pickle.load(f)

    # terminal clear
    os.system("cls")

    print("[Status Monitoring]")
    print("* Torch version:", torch_version)
    print("* CUDA version:", torch_cuda_version)
    print("* Torch CUDA available:", torch_cuda_available)
    print("")

    print("* CPU usage:", psutil.cpu_percent())
    print("* Memory usage:", psutil.virtual_memory().percent)
    print("* Disk usage:", psutil.disk_usage("/").percent)
    print("")


    # os.system("gpustat -c")  # windows에서 안됨..
    print("")

    for i, c in enumerate(c_list):

        print(str(i + 1) + "/" + str(len(c_list)))
        print(c.command_str)

        if c.status == " Done ":
            print(c.end_time)
            print(c.elapsed_time)

            print(Fore.BLACK + Back.GREEN + c.status)
            print(Style.RESET_ALL)

        if c.status == " Running ":
            print(Fore.BLACK + Back.YELLOW + " Running ")
            print(Style.RESET_ALL)

        if c.status == " Waiting ":
            print(Fore.BLACK + Back.RED + " Waiting ")
            print(Style.RESET_ALL)

    time.sleep(2)
