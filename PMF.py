from pylab import *
from dataloader import dataloader
import random
import math


def predict(U, V):
    N, D = U.shape
    M, D = V.shape
    rating_list = []
    for user in range(N):
        user_rating = np.sum(U[user, :] * V, axis=1)
        rating_list.append(user_rating)
    rating_predict = np.array(rating_list)
    return rating_predict


def PMF(train, test, N, M, lr, D, lambda_1, lambda_2, epoch_num):
    U = np.random.normal(0, 0.1, (N, D))  # latent user feature matrix
    V = np.random.normal(0, 0.1, (M, D))  # latent itme feature matrix
    loss_threshold = 1000.0  # threshold of loss (optional)
    rmse_list = []
    loss_list = []
    for epoch in range(epoch_num):
        loss = 0.0
        for data in train:
            user_id, item_id, rating = data
            # SGD update
            error = rating - np.dot(U[user_id], V[item_id].T)
            U[user_id] = U[user_id] + lr * (error * V[item_id] - lambda_1 * U[user_id])
            V[item_id] = V[item_id] + lr * (error * U[user_id] - lambda_2 * V[item_id])

            loss = 0.5 * (error ** 2 + lambda_1 * np.square(U[user_id]).sum() + lambda_2 * np.square(V[item_id]).sum())

        rating_predict = predict(U, V)

        loss_list.append(loss)
        rmse_train = RMSE(U, V, train)
        rmse_test = RMSE(U, V, test)
        rmse_list.append((rmse_train,rmse_test))
        # if np.sum(loss_list, axis = 0) < loss_threshold:
        #     print('Complete!')
        #     break
        if epoch % 10 == 0:
            print('Epoch:{}/{}, Loss:{}, RMSE-train:{}, RMSE-test:{}'.format(epoch, epoch_num, loss, rmse_train, rmse_test))
    print('-'*50)
    print('Result of user rating predict (row is user, column is item):')
    print(rating_predict[0:10])
    return loss_list, rmse_list, U, V


def RMSE(U, V, test):
    count = len(test)
    sum_rmse = 0.0
    for data in test:
        user_id, item_id, rating = data
        predict_rating = np.dot(U[user_id], V[item_id].T)
        sum_rmse += np.square(rating - predict_rating)
    rmse = np.sqrt(sum_rmse / count)
    return rmse


if __name__ == '__main__':
    dir_data = "dataset/u.data"
    np.set_printoptions(precision=2,)
    proportion = 0.8

    # N is the number of user, M is the number of items, train and test are datasets divided by the proportion
    N, M, train, test = dataloader(dir_data, proportion)

    lr = 0.005  # learning rate
    D = 10  # the number of latent feature
    lambda_1 = 0.1  # lambda_1 & lambda_2: regularization parameters
    lambda_2 = 0.1
    epoch_num = 50  # the number of epoch
    loss, rmse, U, V = PMF(train, test, N, M, lr, D, lambda_1, lambda_2, epoch_num)
