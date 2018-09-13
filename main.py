import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as Data
from torchvision.models import resnet18, resnet50
from torch.autograd import Variable
from datasets import ImageDataset
from sklearn.model_selection import train_test_split, KFold
import torchvision.transforms as transforms
from PIL import Image
import os.path as osp
import os

EPOCH = 26
BATCH_SIZE = 64
LR = 1e-4
out_dim = 5
transfrom_nums = 1
use_all_data = True

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
transforms_train_list = [#transforms.Resize((320, 320), interpolation=3), #Image.BICUBIC
                         transforms.RandomHorizontalFlip(),
                         transforms.ToTensor(),
                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])]
transforms_val_list = [#transforms.Resize((320,320), interpolation=3), #Image.BICUBIC
                       transforms.ToTensor(),
                       transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])]
data_transforms = {
    'train': transforms.Compose(transforms_train_list),
    'val': transforms.Compose(transforms_val_list),
    'test':  transforms.Compose(transforms_val_list),
}
root_dir = './raw_data'
use_gpu = torch.cuda.is_available()
print('use_gpu is ', use_gpu)

def merge_data_label(data_file, image_path, info=None):
    # 获取训练集和测试集的数据
    data = pd.read_csv(data_file)
    file_list = data['filename'].values.tolist()
    image_data = np.zeros((len(file_list), 320, 320, 3))
    print('data size is {}'.format(len(image_data.shape)))
    for i, x in enumerate(file_list):
        path = osp.join(image_path, x)
        if osp.isfile(path):
            image = Image.open(path).resize((320, 320)).convert('RGB')
            image_data[i, :, :, :] = np.array(image)

    if info == 'train':
        print('merge train data....')
        train_y = data['type'].values
        np.save('data/train_y.npy', train_y)
        np.save('data/train_x.npy', image_data)
    else:
        np.save('data/test_x.npy', image_data)

def step(model, data_loader, lossFunc, optimizer, mode, data_size=None):
    if mode == 'val':
        model.eval()
        predict = np.zeros((data_size, ))
    for step, (batch_x, batch_y) in enumerate(data_loader):
        if use_gpu:
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
        onehot_predict = model(batch_x)
        loss = lossFunc(onehot_predict, batch_y)
        loss += loss.data[0]
        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            index = step*BATCH_SIZE
            prob = nn.functional.softmax(onehot_predict, dim=1)
            pred_label = torch.argmax(prob, dim=1)
            predict[index: index+batch_x.size(0)] = pred_label

    if data_size == None:
        predict = None

    return loss, predict

def train():
    ##########################################################################
    # Model Generating
    model = resnet18(pretrained=False)
    model.avgpool = nn.AdaptiveAvgPool2d((1, 1)) #即使B*C*M*H -> B*C*1*1
    model.fc = nn.Linear(model.fc.in_features, out_dim)
    if use_gpu:
        model = model.cuda()
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    lossFunc = nn.CrossEntropyLoss()  # the target is not one-hotted
  
    # 读取训练集和测试集
    data_x = np.load('./data/train_x.npy')
    data_y = np.load('./data/train_y.npy')
    test_x = np.load('./data/test_x.npy')
    if use_all_data:
        train_x, train_y = data_x ,data_y
    else:
        train_x, val_x ,train_y, val_y = train_test_split(data_x, data_y, random_state=2018, test_size=0.2)
        val_loader = Data.DataLoader(dataset=ImageDataset(val_x, labels=val_y, transform=data_transforms['val']),
                                     batch_size=BATCH_SIZE,
                                     shuffle=False,
                                     num_workers=15)
    
    train_loader = Data.DataLoader(dataset=ImageDataset(train_x, labels=train_y, transform=data_transforms['train']),
                                   batch_size=BATCH_SIZE,
                                   shuffle=True,
                                   num_workers=15)
    
    test_loader = Data.DataLoader(dataset=ImageDataset(test_x, transform=data_transforms['test']),
                                  batch_size=BATCH_SIZE,
                                  shuffle=False,
                                  num_workers=15)
    ###########################################################################
    # Training and Validating
    for epoch in range(EPOCH):
        # model train
        print('Training...')
        train_loss, _ = step(model, train_loader, lossFunc, optimizer, 'train')
        
        if use_all_data:
            print('Epoch: ', epoch, '|Train Loss: ', train_loss)
        
        else:
            val_loss, predict = step(model, val_loader, lossFunc, optimizer, 'val', val_x.shape[0])
            acc = (predict==val_y).sum() / val_y.shape[0]       
            print('Val Accuracy is {:.4f}'.format(acc))
            print('Epoch: ', epoch, '|Train Loss: ', train_loss, '|Val Loss: ', eval_loss, )

    ############################################################################
    # Evaluate
    model.eval()
    test_predict = np.zeros((test_x.shape[0],))
    for step, batch_x in enumerate(test_loader):
        if use_gpu:
            batch_x = batch_x.cuda()
        onehot_predict = model(batch_x)
        prob = nn.functional.softmax(onehot_predict, dim=1)
        pred_label = torch.argmax(prob, dim=1)
        index = step*BATCH_SIZE
        test_predict[index:index+batch_x.size(0)] = predict_label.cpu().data.numpy()


if __name__ ==  '__main__':
    merge_data_label(osp.join(root_dir, 'train.csv'), osp.join(root_dir, 'train'), 'train')
    merge_data_label(osp.join(test_dir, 'test.csv'), osp.join(root_dir, 'test'))
    train()

