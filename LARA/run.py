import torch.utils.data
from LARA.data.dataloader import dataset
from LARA.train import train
from LARA.test import test

if __name__ == "__main__":

    train_dataset = dataset('data/train/train_data.csv', 'data/train/user_emb.csv')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1024, shuffle=True, num_workers=0)
    neg_dataset = dataset('data/train/neg_data.csv', 'data/train/user_emb.csv')
    neg_loader = torch.utils.data.DataLoader(neg_dataset, batch_size=1024, shuffle=True, num_workers=0)

    # 训练
    model_path, _, g_loss, d_loss = train(train_loader, neg_loader, min(neg_dataset.__len__(), train_dataset.__len__()))

    # 测试
    n_10,p_10 = test(model_path)