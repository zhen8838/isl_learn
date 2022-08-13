import sys
import shutil
import subprocess
import shlex
import os

l = ["01_presburger_sets", "02_iteration-domains", "03_schedules", "04_memory",
     "05_dependences", "06_classical-loop-transformations", "07_ast-generation", "08_c-parser"]

os.chdir("out/")

for f in l:
  shutil.copyfile(f"../{f}.ipynb", f"{f}.ipynb")
  ret = subprocess.run(shlex.split(
      f'jupyter nbconvert {f}.ipynb --to markdown --output {f}.md'))

with open("polyherdal_learn.md", 'w') as of:
  for in_file in l:
    with open(f'{in_file}.md', 'r') as in_f:
      of.writelines(in_f.readlines())
