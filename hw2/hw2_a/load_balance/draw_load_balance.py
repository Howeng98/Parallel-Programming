import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
from subprocess import Popen, PIPE, STDOUT

filename = input("Please input the filename: ")
testcase = input("\nPlease input the testcase: ")

def print_command(p):
  for line in p.stdout:
    print(line.strip())

thread_num = [1, 2, 4, 8, 12]
testcases = {0: "174170376 -0.7894722222222222 -0.7825277777777778 0.145046875 0.148953125 2549 1439",
            1: "10000 -0.29899250664589705 -0.2772002993319328 -0.6327591639095336 -0.6433840614725646 7680 4320",
            2: "10000 -0.5 0.5 -0.5 0.5 491 935"}

testcase = testcases[int(testcase)]

p = Popen(["make clean"], shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True)
print_command(p)

p = Popen(["make"], shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True)
print_command(p)

average = [0.922019, 0.868065]
highest = [0.927443, 0.869428]
lowest = [0.91135, 0.859344]
labels = ['Pthread', 'Hybrid']

X = np.arange(2)
Y = np.arange(0.8, 0.96, 0.02)

# bar chart for different thread number's CPU, COMM, IO time

fig, ax = plt.subplots()
ax.bar(X-0.25, average, width=0.22, label='Average Time')
ax.bar(X, highest, width=0.22, label='Highest Time')
ax.bar(X+0.25, lowest, width=0.22, label='Lowest Time')
ax.set_xticks(X)
ax.set_xticklabels(labels)
plt.ylim(0.8, 0.96)
ax.set_yticks(Y)
ax.set_ylabel("Runtime (seconds)")
ax.set_title(f"Time Profile")
ax.legend(loc="upper right")
fig.savefig(f"./images/{filename}_time_profile_bar.png")
