import os
import numpy as np
log_dir = os.getcwd()+'/log_qbc'
task_list = ["halfcheetah", "walker2d", "hopper"]
data_list = ["-random-v2","-medium-v2", "-medium-replay-v2","-medium-expert-v2"]

file_names = os.listdir(log_dir)
total_name, result = [], []
for data in data_list:
  for task in task_list:
    tmp_np = []
    for file_name in file_names:
      if task+data in file_name:
        file = open(log_dir+'/'+file_name)
        while True:
          line = file.readline()
          line_list = line.split()
          if len(line_list) > 2:
              tmp_np.append(np.array([float(line_list[2])]))
          if not line:
            break

    total_name.append(task + data)
    result.append(np.array(tmp_np))

for idx, name in enumerate(total_name):
  print("[TASK] : "+'%30s'%name+"   [Mean] : "+'%8s'%str(round(np.mean(result[idx]),2)) +"   [STD] : "+str(round(np.std(result[idx]),2)))

