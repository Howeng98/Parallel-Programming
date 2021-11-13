import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
from subprocess import Popen, PIPE, STDOUT

n = input("please input n: ")
testcase = input("please input testcase: ")

def print_command(p):
  for line in p.stdout:
    print(line.strip())

process_num = [pow(2, i) for i in range(6)]

p = Popen(["make clean"], shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True)
print_command(p)

p = Popen(["make"], shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True)
print_command(p)

result_cpu = list()
result_comm = list()
result_io = list()

for process in tqdm(process_num):
  N = int(process / 12) + 1
  cmd = f"srun -N{N} -n{process} ./hw1 {n} {testcase}.in {testcase}.out"
  p = Popen(cmd, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True)
  line = [i for i in p.stdout]
  result_cpu.append(float(line[0]))
  result_comm.append(float(line[1]))
  result_io.append(float(line[2]))

labels = [str(i) for i in process_num]

# bar chart for different process number's CPU, COMM, IO time
fig, ax = plt.subplots()
ax.bar(labels, result_cpu, label='CPU_time')
ax.bar(labels, result_comm, bottom=result_cpu, label='COMM_time')
ax.bar(labels, result_io, bottom=np.array(result_cpu)+np.array(result_comm), label='IO_time')
ax.set_xlabel("process number")
ax.set_ylabel("runtime (seconds)")
ax.set_title(f"testcase{testcase} performance")
ax.legend(loc="upper right")
fig.savefig(f"./images/{testcase}_bar.png")

# line chart for speed up factor
sum = [result_cpu[i]+result_comm[i]+result_io[i] for i in range(len(process_num))]
sum = [sum[0]/sum[i] for i in range(len(process_num))]
plt.figure(dpi=100, linewidth=2)
plt.plot(process_num, sum, 'o-', color='g')
plt.xlabel('SpeedUp Factor')
plt.xticks(np.arange(0, 33, 5))
plt.ylabel('process number')
plt.title(f"testcase{testcase} speedup factor")
plt.savefig(f"./images/{testcase}_line.png")