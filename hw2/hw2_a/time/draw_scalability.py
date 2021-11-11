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

thread_num = [2, 4, 6, 8, 10, 12]
testcases = {0: "174170376 -0.7894722222222222 -0.7825277777777778 0.145046875 0.148953125 2549 1439",
            1: "10000 -0.29899250664589705 -0.2772002993319328 -0.6327591639095336 -0.6433840614725646 7680 4320"}

testcase = testcases[int(testcase)]

p = Popen(["make clean"], shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True)
print_command(p)

p = Popen(["make"], shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True)
print_command(p)

result_cpu = list()
result_comm = list()
result_io = list()

for thread in tqdm(thread_num):
  cmd = f"srun -c{thread} -n1 ./hw2a experiment.png {testcase}"
  p = Popen(cmd, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True)
  line = [i for i in p.stdout]
  result_cpu.append(float(line[0]))
  result_comm.append(float(line[1]))
  result_io.append(float(line[2]))

labels = [str(i) for i in thread_num]

# bar chart for different thread number's CPU, COMM, IO time
fig, ax = plt.subplots()
ax.bar(labels, result_cpu, label='CPU_time')
ax.set_xlabel("Thread Number")
ax.set_ylabel("Runtime (seconds)")
ax.set_title(f"Time Profile")
ax.legend(loc="upper right")
fig.savefig(f"./images/{filename}_time_profile_bar.png")

# line chart for different thread number's CPU, COMM, IO time
fig, ax = plt.subplots()
ax.plot(labels, result_cpu, label='CPU_time')
ax.set_xlabel("Thread Number")
ax.set_ylabel("Runtime (seconds)")
ax.set_title(f"Time Profile")
ax.legend(loc="upper right")
fig.savefig(f"./images/{filename}_time_profile_line.png")


# line seperate speedup factor
cpu_speedup = [result_cpu[0]/result_cpu[i] for i in range(len(thread_num))]
fig, ax = plt.subplots()
ax.plot(labels, cpu_speedup, label='CPU')
ax.set_xlabel("Thread Number")
ax.set_ylabel("Speedup")
ax.set_title(f"Seperate Speedup")
ax.legend(loc="upper right")
fig.savefig(f"./images/{filename}_speedup.png")