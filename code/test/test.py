import sys
print(sys.path)
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
print(sys.path)
import FINDER_ND as FINDER
# from GraphSpectualDM.parsers.config import get_config
# from GraphSpectualDM.generate_graphs import generate_graphs_random
import subprocess

subprocess.run('conda run -n torch_py38 python ../GraphSpectualDM/generate_graphs.py', shell=True)



# # 虚拟环境1中的Python解释器路径
# virtualenv1_python = "/path/to/virtualenv1/bin/python"
# # 运行子进程
# subprocess.run([virtualenv1_python, "script.py"])

# get_config("Digg", 12)
# generate_graphs_random()








