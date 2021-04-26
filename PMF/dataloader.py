import numpy as np

def dataloader(dataset, proportion):
    user_set = {}
    item_set = {}
    user_number = 0
    item_number = 0
    data = []
    f = open(dataset)
    for line in f.readlines():
        user_id, item_id, rating, timestamp = line.split()
        if int(user_id) not in user_set:
            user_set[int(user_id)] = user_number
            user_number += 1
        if int(item_id) not in item_set:
            item_set[int(item_id)] = item_number
            item_number += 1
        data.append([user_set[int(user_id)], item_set[int(item_id)], int(rating)])
    f.close()
    N = user_number;
    M = item_number;

    np.random.shuffle(data)
    train = data[0:int(len(data) * proportion)]
    test = data[int(len(data) * proportion):]
    return N, M, train, test
