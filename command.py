import pickle
import os
import time
import datetime
# from pushbullet import Pushbullet
import getpass


class command():
    def __init__(self, command_str, status, end_time, elapsed_time):
        self.command_str = command_str
        self.status = status
        self.end_time = end_time
        self.elapsed_time = elapsed_time


command_list = [
###

## Done
# "python train.py --yml train_config",  # timm_tf_efficientnetv2_m, bathsize 16 => pub test: 97.19

## Progress
"python train.py --yml train_config",  # timm_tf_efficientnetv2_m, bathsize 16 => pub test: 97.19


## not yet
# "python train.py --yml train_config1",  # tf_efficientnet_l2_ns_475, bathsize 2

# "python train.py --yml train_config2",  # swin_large_patch4_window12_384, bathsize 16

# "python train.py --yml train_config3",  # vit_large_patch16_384, bathsize 32

# "python train.py --yml train_config4",  # cait_m48_448, bathsize 32

# "python train.py --yml train_config5",  # cait_m48_448, bathsize 32

# "python train.py --yml train_config5",  # tf_efficientnetv2_l_in21ft1k, bathsize 32

# "python train.py --yml train_config5",  # tf_efficientnetv2_m_in21ft1k, bathsize 32


###
]

# initially, generate command objects.
c_list = []
for command_elem in command_list:
    c_i = command(command_elem, " Waiting ", "", "")
    c_list.append(c_i)


# run commands.
for i, c_i in enumerate(c_list):
    # start of command
    print("")
    print("Command: {}/{}".format((i + 1), len(c_list)))
    print("start_time:", str(datetime.datetime.now()))
    print(c_i.command_str)
    print("")

    # logging
    c_i.status = " Running "
    with open("log.pkl", "wb") as f:
        pickle.dump(c_list, f)

    start_time = time.time()
    os.system(c_i.command_str)
    ##
    time.sleep(0.1)
    ##
    end_time = time.time()

    # end of command
    print("")
    print("end_time:", str(datetime.datetime.now()))

    hours, rem = divmod(end_time - start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))

    # logging
    c_i.status = " Done "
    c_i.end_time = "End time: " + str(datetime.datetime.now())
    c_i.elapsed_time = "Elapsed time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)
    with open("log.pkl", "wb") as f:
        pickle.dump(c_list, f)

