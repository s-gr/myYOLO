import re

def get_labels(path_to_dir):

    train_set = {}
    my_list_temp = []

    with open(path_to_dir, 'r') as file:
        my_str_arr = file.readlines()

    for line in range(len(my_str_arr)):

        if 'path' in my_str_arr[line]:
            match = re.search('([^/]*(?=.png$))', my_str_arr[line])
            if match:
                match.group()
                train_set.update({match.group(): my_list_temp})
                # print(match.group() + ': ' + str(train_set[match.group()]))
            else:
                train_set.update({'None': my_list_temp})
                print('None' + str(my_list_temp))
            my_list_temp = []

        elif 'label' in my_str_arr[line]:
            temp_var = my_str_arr[line]

            if '}' not in my_str_arr[line]:
                temp_var += my_str_arr[line + 1]

            temp_var = temp_var.replace(',', '').replace('- {label:', '').replace(':', '').replace('occluded', '').replace(
                'x_max', '').replace('x_min', '').replace('y_min', '').replace('y_max', '').replace('}', '').replace("'", '')
            temp_var = temp_var.split()

            for i in range(1, len(temp_var)):
                if i == 1:
                    temp_var[i] = int(temp_var[i] == 'true')
                else:
                    temp_var[i] = float(temp_var[i])

            my_list_temp += temp_var

    return train_set

