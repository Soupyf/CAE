import random
import numpy as np
from Model import Encoder, Decoder
import torch
from torch import autograd
import torch.nn as nn
from torch.utils.data import random_split, DataLoader
from createDataLoader import data
from torch import optim
import matplotlib.pyplot as plt
from torchvision import transforms
import torchvision.utils


# torch.autograd.set_detect_anomaly(True)
epoch = 200
encoder = Encoder()
decoder = Decoder()
encoder = encoder.to('cuda')
decoder = decoder.to('cuda')
# transform = transforms.Compose([
#     transforms.RandomHorizontalFlip(),
#     transforms.Normalize(
#         mean=[0.485, 0.456, 0.406],
#         std=[0.229, 0.224, 0.225],
#     )
# ])
data = data()
print(len(data))
train_set, test_set = random_split(dataset=data,
                                   lengths=[5000, 579],
                                   generator=torch.Generator().manual_seed(random.randint(0, 10000)))
train_loader = DataLoader(train_set, 256, True)
test_loader = DataLoader(test_set, 256, True)
criterion_en = nn.MSELoss()
criterion_de = nn.MSELoss()
optimizer_en = optim.Adam(encoder.parameters(), lr=0.001, weight_decay=0.00002)
optimizer_de = optim.Adam(decoder.parameters(), lr=0.001, weight_decay=0.00002)
train_loss = []
test_loss = []
encoder_train_loss = []
decoder_train_loss = []
encoder_test_loss = []
decoder_test_loss = []
tensor=[]
# decoder.load_state_dict(torch.load("decode.pth"))

for i, (feature, label) in enumerate(test_loader):
    feature = feature.to('cuda')
    tensor.append(decoder(feature))

# for i in range(len(tensor)):
#     t = tensor[i][0]
#     torchvision.utils.save_image(t, 'img/' + str(i) + '.png')



# 先单独训练encoder
for e in range(1, epoch + 1):
    loss = []
    loss_all = []
    for i, (feature, label) in enumerate(train_loader):
        feature = feature.to('cuda')
        label = label.to('cuda')
        e_pred = encoder(label)
        loss1 = criterion_en(e_pred, feature)
        loss.append(loss1)
        loss1 = loss1.requires_grad_()
        optimizer_en.zero_grad()
        loss1.backward()
        optimizer_en.step()
    loss = torch.tensor(loss)
    encoder_train_loss.append(torch.mean(loss))
    print("epoch {}, loss_encoder: {}".format(e, torch.mean(loss)))

    for i, (feature, label) in enumerate(train_loader):
        feature = feature.to('cuda')
        label = label.to('cuda')
        e_pred = encoder(label)
        d_pred = decoder(e_pred)
        loss2 = criterion_de(d_pred, label)
        loss_all.append(loss2)
        loss2 = loss2.requires_grad_()
        optimizer_de.zero_grad()
        loss2.backward()
        optimizer_de.step()
    loss_all = torch.tensor(loss_all)
    decoder_train_loss.append(torch.mean(loss_all))
    print("epoch {}, loss_decoder: {}".format(e, torch.mean(loss_all)))
    # 利用训练好的

    tl1 = []
    tl2 = []
    for i, (feature, label) in enumerate(test_loader):
        feature = feature.to('cuda')
        label = label.to('cuda')
        e_predt = encoder(label)
        d_predt = decoder(e_predt)
        l1 = criterion_en(e_predt, feature)
        l2 = criterion_de(d_predt, label)
        tl1.append(l1)
        tl2.append(l2)
    tl1 = torch.tensor(tl1)
    tl2 = torch.tensor(tl2)
    encoder_test_loss.append(torch.mean(tl1))
    decoder_test_loss.append(torch.mean(tl2))
    test_loss.append(torch.mean(tl1) + torch.mean(tl2))
    print("epoch {}, test_loss: {}".format(e, torch.mean(tl1) + torch.mean(tl2)))
    if e == 50:
        torch.save(decoder.state_dict(), 'decode50.pth')
    if e == 100:
        torch.save(decoder.state_dict(), 'decode100.pth')
x = range(epoch)
for i in x:
    train_loss.append(encoder_train_loss[i] + decoder_train_loss[i])
fig, axes = plt.subplots(figsize=(10, 10))  # 创建一个图形对象和一个子图对象

axes.plot(x, encoder_train_loss, label='encoder_train')
axes.plot(x, decoder_train_loss, label='decoder_train')
axes.plot(x, train_loss, label='train')
axes.plot(x, encoder_test_loss, label='encoder_test')
axes.plot(x, decoder_test_loss, label='decoder_test')
axes.plot(x, test_loss, label='test')
axes.legend()
plt.savefig("loss")
plt.show()
