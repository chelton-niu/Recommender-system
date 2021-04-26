import torch
from torch import nn




attr_num = 18  # item属性数
attr_present_dim = 5  # 属性维度
hidden_dim = 100  # G隐藏层维度
user_emb_dim = attr_num  # user维度与item一致
attr_dict_size = 2 * attr_num  # 原始数据0~35 因此词典大小:2*attr_num



# 生成器
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.G_attr_matrix = nn.Embedding(attr_dict_size, attr_present_dim)
        # 正态分布
        for i in self.G_attr_matrix.modules():
            nn.init.xavier_normal_(i.weight)

        # 输入:attr_num*attr_present_dim
        # 输出:hidden_dim
        self.l1 = nn.Linear(attr_num * attr_present_dim, hidden_dim, bias=True)
        self.l2 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.l3 = nn.Linear(hidden_dim, user_emb_dim, bias=True)
        self.af = nn.Tanh()
        for i in self.modules():
            if isinstance(i, nn.Linear):
                nn.init.xavier_normal_(i.weight)
                nn.init.xavier_normal_(i.bias.unsqueeze(0))
            else:
                pass

    def forward(self, attribute_id):
        attribute_id = attribute_id.long()
        attr_present = self.G_attr_matrix(attribute_id)
        attr_feature = torch.reshape(attr_present, [-1, attr_num * attr_present_dim])

        # 输入:attr_f [batch_size, attr_num*attr_present_dim]
        # 输出:user_g [batch_size, user_emb_dim]
        o1 = self.af(self.l1(attr_feature))
        o2 = self.af(self.l2(o1))
        o3 = self.af(self.l3(o2))
        return o3


# 判别器
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.D_attr_matrix = nn.Embedding(attr_dict_size, attr_present_dim)
        for i in self.D_attr_matrix.modules():
            nn.init.xavier_normal_(i.weight)
        self.l1 = nn.Linear(attr_num * attr_present_dim + user_emb_dim, hidden_dim, bias=True)
        self.l2 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.l3 = nn.Linear(hidden_dim, user_emb_dim, bias=True)
        self.af = nn.Tanh()
        for i in self.modules():
            if isinstance(i, nn.Linear):
                nn.init.xavier_normal_(i.weight)
                nn.init.xavier_normal_(i.bias.unsqueeze(0))
            else:
                pass

    def forward(self, attribute_id, user_emb):
        attribute_id = attribute_id.long()
        attr_present = self.D_attr_matrix(attribute_id)
        attr_feature = torch.reshape(attr_present, [-1, attr_num * attr_present_dim])
        # 连接item向量和user向量 （用户-物品对）
        user_item_emb = torch.cat((attr_feature, user_emb.float()), 1)
        user_item_emb = user_item_emb.float()

        # 输入:user_item_emb[batch_size, attr_num*attr_present_dim + user_emb_dim]
        # 输出:[batch_size, user_emb_dim]
        o1 = self.af(self.l1(user_item_emb))
        o2 = self.af(self.l2(o1))
        o3 = self.l3(o2)
        y_prob = torch.sigmoid(o3)  # 概率得分
        return y_prob




