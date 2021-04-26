import torch,time
from tqdm import tqdm
import torch.nn as nn
from .data.model import Generator, Discriminator
from .data import dataloader

learning_rate = 0.0001
epoch = 50
batch_size = 1024
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
g_loss_records = []
d_loss_records = []
g_path = []
d_path = []


def train(train_data, neg_data, minLen):
    start_time = time.time()
    print("Training on", device)
    gen = Generator()
    dis = Discriminator()
    g_optimizer = torch.optim.Adam(gen.parameters(), lr=learning_rate, weight_decay=0)
    d_optimizer = torch.optim.Adam(dis.parameters(), lr=learning_rate, weight_decay=0)
    loss = nn.BCELoss(reduction='mean')  # 平均loss 交叉熵
    gen = gen.to(device)
    dis = dis.to(device)

    for i in tqdm(range(epoch)):
        # D
        neg_iter = neg_data.__iter__()
        temp = 0
        for user, item, attr, user_emb in train_data:
            if batch_size * temp >= minLen:
                break
            # 正例
            attr = attr.to(device)
            user_emb = user_emb.to(device)
            # 负例
            neg_user, neg_item, neg_attr, neg_user_emb = neg_iter.next()
            neg_attr = neg_attr.to(device)
            neg_user_emb = neg_user_emb.to(device)
            # 生成
            gen_pos_user_emb = gen(attr)
            gen_neg_user_emb = gen(neg_attr)
            gen_pos_user_emb = gen_pos_user_emb.to(device)
            gen_neg_user_emb = gen_neg_user_emb.to(device)
            # D loss
            y_pos_d = dis(attr, user_emb)
            y_neg_d = dis(neg_attr, neg_user_emb)
            y_gen_pos_d = dis(attr, gen_pos_user_emb)
            y_gen_neg_d = dis(neg_attr, gen_neg_user_emb)
            d_optimizer.zero_grad()
            d_loss_pos = loss(y_pos_d, torch.ones_like(y_pos_d))  # 正例 1
            d_loss_neg = loss(y_neg_d, torch.zeros_like(y_neg_d))  # 负例 0
            d_loss_gen_pos = loss(y_gen_pos_d, torch.zeros_like(y_gen_pos_d))  # 生成 0
            d_loss_gen_neg = loss(y_gen_neg_d, torch.zeros_like(y_gen_neg_d))  # 生成 0
            d_loss = torch.mean(d_loss_pos + d_loss_neg + d_loss_gen_pos + d_loss_gen_neg)
            d_loss.backward()  # 反向传播，计算当前梯度
            d_optimizer.step()  # 根据梯度更新网络参数
            temp += 1

        # G
        for user, item, attr, user_emb in train_data:
            # G loss
            g_optimizer.zero_grad()
            attr = attr.long()
            attr = attr.to(device)
            gen_user_emb = gen(attr)
            gen_user_emb = gen_user_emb.to(device)
            y_gen = dis(attr, gen_user_emb)
            y_true = torch.ones_like(y_gen)
            g_loss = loss(y_gen, y_true)
            g_loss.backward()
            g_optimizer.step()

        g_loss_records.append(g_loss.item())
        d_loss_records.append(d_loss.item())

        total_time = time.time() - start_time
        print("EPOCH",i+1," | D_LOSS:", d_loss.item(),
              ",G_LOSS:",g_loss.item())
        if i % 10 == 0:
            torch.save(gen.state_dict(), 'weight/g_' + str(i) + ".pt")
            torch.save(dis.state_dict(), 'weight/d_' + str(i) + ".pt")
            g_path.append('weight/g_' + str(i) + ".pt")
            d_path.append('weight/d_' + str(i) + ".pt")

    return g_path, d_path, g_loss_records, d_loss_records
