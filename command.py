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

"python train.py --yml train_config",

"python train.py --yml train_config1",

"python train.py --yml train_config2",

"python train.py --yml train_config3",

"python train.py --yml train_config4",

"python train.py --yml train_config5",

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

