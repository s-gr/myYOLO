from my_utils import get_labels

my_dict = get_labels('dataset/train/train.yaml')

my_str='207384'
print(my_str + ': ' + str(len(my_dict[my_str]) // 6))
