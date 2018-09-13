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

#1104
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

"""
attribute_list.txt: (30, 2) ->(index, attribute_name)
attributes_per_class: (-1, 31) ->(label, attribute30)
class_embedding: (-1, 301) -> (name, embedding300)
label_list: (230, 2) -> (label, name)
"""

def merge_data_label(data_file, image_path, info=None):
    # 获取训练集和测试集的数据
    data = pd.read_csv(data_file)
    file_list = data['filename'].values.tolist()
    image_data = np.zeros((len(file_list), 320, 320, 3))
    print('data size is {}'.format(len(image_data.shape)))
    for i, x in enumerate(file_list):
        path = osp.join(image_path, x)
        if osp.isfile(path) and ('gif' not in path):
            image = Image.open(path).resize((320, 320)).convert('RGB')
            image_data[i, :, :, :] = np.array(image)

    if info == 'train':
        print('merge train data....')
        train_y = data['type'].values
        np.save('data/train_y.npy', train_y)
        np.save('data/train_data.npy', image_data)
    else:
        np.save('data/test_data.npy', image_data)

def train():
    model = resnet18(pretrained=False)
    model.avgpool = nn.AdaptiveAvgPool2d((1, 1)) #即使B*C*M*H -> B*C*1*1
    model.fc = nn.Linear(model.fc.in_features, out_dim)
    #model = models.init_model('resnet18', out_dim)
    if use_gpu:
        model = model.cuda()
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    lossFunc = nn.CrossEntropyLoss()  # the target is not one-hotted
  
    # 读取训练集和测试集
    data_x = np.load('./data/train_data.npy')
    data_y = np.load('./data/train_y.npy')
    #data_y = normalize(data_y, 'l2')
    print(data_x.shape)
    print(data_y.shape)
    #test_x = np.load('./exp/test_data.npy')
    if use_all_data:
        train_x, train_y = data_x ,data_y
    else:
        kf = KFold(n_splits=5, random_state=2018).split(data_x)
        for idx, (train_index, val_index) in enumerate(kf):
            train_x, train_y = data_x[train_index], data_y[train_index]
            val_x, val_y = data_x[val_index], data_y[val_index]
            break
        #train_x, val_x ,train_y, val_y = train_test_split(data_x, data_y, random_state=2018, test_size=0.2)
        #划分训练集和验证集
        print('After split train size is {}, val_size is {}'.format(train_x.shape, val_x.shape))
        
        val_loader = Data.DataLoader(dataset=ImageDataset(val_x,
                                                       labels=val_y,
                                                       transform=data_transforms['val']),
                                     batch_size=BATCH_SIZE,
                                     shuffle=False,
                                     num_workers=15)
#

    
    train_loader = Data.DataLoader(dataset=ImageDataset(train_x,
                                                        labels=train_y,
                                                        transform=data_transforms['train']),
                                   batch_size=BATCH_SIZE,
                                   shuffle=True,
                                   num_workers=15)
    
    #test_loader = Data.DataLoader(dataset=ImageDataset(test_x,
    #                                                   transform=data_transforms['test']),
    #                              batch_size=BATCH_SIZE,
    #                              shuffle=False,
    #                              num_workers=15)

    for epoch in range(EPOCH):
        train_loss = 0.
        # model train
        print('Training...')
        train_predict = np.zeros((train_x.shape[0], out_dim))
        if not use_all_data:
            val_predict = np.zeros((val_x.shape[0], out_dim))
        #test_predict = np.zeros((test_x.shape[0], out_dim))
        for step, (batch_x, batch_y) in enumerate(train_loader):
            if use_gpu:
                batch_x = batch_x.cuda()
                batch_y = batch_y.cuda()
            predict = model(batch_x)
            loss = lossFunc(predict, batch_y)
            train_loss += loss.data[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            index = step * BATCH_SIZE
            train_predict[index:index + batch_x.size(0)] = predict.cpu().data.numpy()
            if step % 50 == 0:
                print('Train elapsed {} steps'.format(step))

        if not use_all_data:
            #model eval
            model.eval()
            eval_loss = 0.
            print('Validation....')
            for step, (batch_x, batch_y) in enumerate(val_loader):
                if use_gpu:
                    batch_x = batch_x.cuda()
                    batch_y = batch_y.cuda()
                onehot_predict = model(batch_x)
                loss = lossFunc(onehot_predict, batch_y)
                prob = nn.functional.softmax(onehot_predict, dim=1)
                pred_label = torch.argmax(prob, dim=1)
                eval_loss += loss.data[0]
                index = step * BATCH_SIZE
                val_predict[index:index + batch_x.size(0)] = predict.cpu().data.numpy()
                if step % 50 == 0:
                    acc = (pred_label==batch_y).sum().numpy()/pred_label.size()[0]
                    print('Step {}: Acc is {:.4f}'.format(step, acc))

        print('-'*100)
        if use_all_data:
            print('Epoch: ', epoch, '|Train Loss: ', train_loss)
        else:
            print('Epoch: ', epoch, '|Train Loss: ', train_loss, '|Val Loss: ', eval_loss, )
        print('-'*100)

        #print('Testing....')
        #model.eval()
        #for step, batch_x in enumerate(test_loader):
        #    if use_gpu:
        #        batch_x = batch_x.cuda()
        #    onehot_predict = model(batch_x)
        #    prob = nn.functional.softmax(onehot_predict, dim=1)
        #    pred_label = torch.argmax(prob, dim=1)
        #    index = step*BATCH_SIZE
        #    test_predict[index:index+batch_x.size(0), :] = predict_label.cpu().data.numpy()
        
        # submit

#if __name__ ==  '__main__':
merge_data_label(osp.join(root_dir, 'train.csv'), osp.join(root_dir, 'train'), 'train')
#merge_data_label(osp.join(test_dir, 'test.csv'), osp.join(root_dir, 'test'))
train()

