import torch,time
import numpy as np
import pandas as pd
from .data.model import Generator, Discriminator
from .data.dataloader import load_test_data
from .data import evaluate

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# 生成推荐用户
def recommendation(gen_user, k):
    # print(gen_user)
    gen_user = gen_user.to(device)
    user_attribute_matrix = pd.read_csv('test/user_attribute.csv', header=None)
    user_attribute_matrix = torch.tensor(np.array(user_attribute_matrix[:]), dtype=torch.float)
    user_embed_matrix = user_attribute_matrix.to(device)
    similar_matrix = torch.matmul(gen_user, user_embed_matrix.T)  #
    index = torch.argsort(-similar_matrix)
    torch.set_printoptions(profile="full")
    rec_users = index[:, 0:k]
    return rec_users


def test(model_path):
    print("+++++++++++++++++++++ Testing ++++++++++++++++++++")
    n_10s = []
    n_20s = []
    p_10s = []
    p_20s = []
    for i in model_path:
        start = time.time()
        print("【Model File】", i)
        # 加载模型
        g = Generator()
        g.load_state_dict(torch.load(i))
        item, attr = load_test_data()
        # print(item)
        item = torch.tensor(item)
        attr = torch.tensor(attr, dtype=torch.long)
        # print(item)
        item = item.to(device)
        attr = attr.to(device)
        gen_user = g(attr)
        rec_user = recommendation(gen_user, 10)
        n_10, p_10 = evaluate(item, rec_user)
        ttime =  time.time() - start
        print("time:", ttime, ",n_10:", n_10, ",p_10:", p_10)
        n_10s.append(n_10)
        # n_20s.append(n_20)
        p_10s.append(p_10)
        # p_20s.append(p_20)
    return n_10s,  p_10s
