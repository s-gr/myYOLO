import os
from yolo_utils import read_classes



with open('dataset/train/train.yaml', 'r') as file:
    my_string = file.readlines()

print(len(my_string))

for line in range(len(my_string)):
    print(my_string[line])
    if 'label' in my_string[line]:
        if '}' not in my_string[line]:


# for line in my_string:
#     if 'label:' in line:
#         line = line.replace(',', '').replace('- {label:', '').replace(':', '')
#         print(line.split())
