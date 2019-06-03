import sys
import fileinput
from sys import argv

run, filename = argv
for line in fileinput.input([filename], inplace=True):
    if line.strip().startswith('%'):
        line = ' ' + line[1:]
    sys.stdout.write(line)

print("completed")

import fileinput
from sys import argv

model = 0
run, filename = argv
with fileinput.FileInput(filename, inplace=True, backup='.bak') as file:
    for line in file:
        if '          15' in line:
            model += 1
            layer = 0
        if '7.5000e+01' in line:
            line.replace('7.5000e+01', 'CHOKOLADE' + layer)
            layer += 1

print(model)