import os
import torch
import argparse
import numpy as np
# import pandas as pd
import torch.utils.data as Data
from utils.binary import assd
from distutils.version import LooseVersion

from Datasets.ISIC2018 import ISIC2018_dataset
from utils.transform import ISIC2018_transform

from Models.networks.ADAMNet import ADAMNet_

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from torch.nn import functional as F
from time import *
import visdom
import matplotlib

os.environ['CUDA_VISIBLE_DEVICES'] = '5'
Test_Model = {'ADAMNet_':  ADAMNet_}

Test_Dataset = {'ISIC2018': ISIC2018_dataset}

Test_Transform = {'ISIC2018': ISIC2018_transform}

vis = visdom.Visdom(env=' ADAMNet_Val')
def Test_isic(test_loader, model):
    isic_dice = []
    isic_iou = []
    isic_assd = []
    Dice = []
    Acc = []
    Jaccard = []
    infer_time = []

    model.eval()
    for step, (img, lab) in enumerate(test_loader):
        image = img.float().cuda()   #shape of image is [1,8,448,448]  torch.float32
        target = lab.float().cuda()   #shape of image is [1,1,448,448]  torch.float32

        # begin_time = time()
        output = model(image)  #output shape [1,2,192,256] torch.float32
        # end_time = time()
        # pred_time = end_time - begin_time
        # infer_time.append(pred_time)


        output_change = F.log_softmax(output,dim=1)
        output_pre = torch.argmax(torch.exp(output_change), dim=1)# shape [1,448,448]
        target = target.squeeze(dim=1).long()#shape [1,448,448]

        output_dis = output_pre.unsqueeze(dim=1)# shape [1,1,448,448]
        label_vis = target.unsqueeze(dim=1)# shape [1,1,448,448]
        vis.images(output_dis, win='val_predict', opts=dict(title='val_predict'))
        vis.images(label_vis, win='val_label', opts=dict(title='val_label'))

        predict_save = output_dis.squeeze().data.cpu().numpy().astype(np.uint8)
        label_save = label_vis.squeeze().data.cpu().numpy().astype(np.uint8)


        f1 = f1_score(label_save.flatten(), predict_save.flatten(), average='binary')
        print('the f1-score of img:', f1)

        acc_score = accuracy_score(label_save.flatten(), predict_save.flatten())
        print('The accuracy of processed img:', acc_score)
        TP = ((predict_save==1)*(label_save==1)).sum()
        FP = (predict_save==1).sum() - TP
        FN = (label_save==1).sum() - TP
        TN = ((predict_save==0)*(label_save==0)).sum()
        precison = TP / (TP + FP)
        recall = TP / (TP + FN)
        sen = TP/(TP + FN)
        spe = TN/(TN + FP)
        # f1_cal = 2*precison*recall / (precison + recall)
        # print('the f1-score of img:', f1_cal)

        AUC = roc_auc_score(label_save.flatten(), predict_save.flatten(), average='macro')
        print('the AUC of img:', AUC)

        print('the Sensitivity of img:', sen)

        print('the Specifity of img', spe)


        # print('The acc of processed img:', acc)

        # predict_save = predict.squeeze().data.cpu().numpy().astype(np.uint8)
        # label_save = label.squeeze().data.cpu().numpy().astype(np.uint8)
        predict_fuse = (np.expand_dims(predict_save, axis=2))
        label_fuse = (np.expand_dims(label_save, axis=2))
        channel_fuse = np.zeros_like(predict_fuse)
        color_label = np.concatenate((255*predict_fuse, 255*label_fuse, channel_fuse), axis=2)
        matplotlib.image.imsave('./result/ADAMNet/color_{}.png'.format(step), color_label)
        matplotlib.image.imsave('./result/predict/predict_{}.png'.format(step), predict_save,cmap='gray')
        matplotlib.image.imsave('./result/label/label_{}.png'.format(step), label_save, cmap='gray')


if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), 'PyTorch>=0.4.0 is required'

    parser = argparse.ArgumentParser(description='U-net add Attention mechanism for biomedical Dataset')
    # Model related arguments
    parser.add_argument('--id', default='ADAMNet_',
                        help='a name for identitying the model. Choose from the following options: Unet_fetus')
    # Path related arguments
    parser.add_argument('--root_path', default='/home/datacenter/ssd2/data_dadong/Skin_dataset',
                        help='root directory of data')
    parser.add_argument('--ckpt', default='./saved_models',
                        help='folder to output checkpoints')
    parser.add_argument('--save', default='./result',
                        help='folder to outoput result')
    parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 16)')
    parser.add_argument('--num_classes', default=2, type=int,
                        help='number of classes')
    parser.add_argument('--num_input', default=8, type=int,
                        help='number of input image for each patient')
    parser.add_argument('--epoch', type=int, default=100, metavar='N',
                        help='choose the specific epoch checkpoints')

    # other arguments
    parser.add_argument('--level', type=int, default=5, help='the number of level')
    parser.add_argument('--mode', default='test', help='training mode')
    parser.add_argument('--data', default='ISIC2018', help='choose the dataset')
    parser.add_argument('--out_size', default=(448, 448), help='the output image size')
    parser.add_argument('--att_pos', default='dec', type=str,
                        help='where attention to plug in (enc, dec, enc\&dec)')
    parser.add_argument('--view', default='axial', type=str,
                        help='use what views data to test (for fetal MRI)')
    parser.add_argument('--val_folder', default='folder1', type=str,
                        help='which cross validation folder')

    args = parser.parse_args()
    args.ckpt = os.path.join(args.ckpt, args.data, args.val_folder, args.id)

    # loading the dataset
    print('loading the {0} dataset ...'.format('test'))
    testset = Test_Dataset[args.data](dataset_folder=args.root_path, folder=args.val_folder,
                                      train_type='test', transform=Test_Transform[args.data])
    testloader = Data.DataLoader(dataset=testset, batch_size=args.batch_size, shuffle=False)
    print('Loading is done\n')

    # Define model
    if torch.cuda.is_available():
        print('We can use', torch.cuda.device_count(), 'GPUs to train the network')
        # if args.data == 'Fetus':
        #     args.num_input = 1
        #     args.num_classes = 3
        #     args.out_size = (256, 256)
        # elif args.data == 'ISIC2018':
        #     args.num_input = 3
        #     args.num_classes = 2
        #     args.out_size = (256, 192)
        model = Test_Model[args.id](args, args.num_input, args.num_classes).cuda()
        # model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))

    # Load the trained best model
    modelname = args.ckpt + '/' + 'min_loss' + '_' + args.data + '_checkpoint.pth.tar'
    if os.path.isfile(modelname):
        print("=> Loading checkpoint '{}'".format(modelname))
        checkpoint = torch.load(modelname)
        model.load_state_dict(checkpoint['state_dict'])
        # optimizer.load_state_dict(checkpoint['opt_dict'])
        print("=> Loaded saved the best model at (epoch {})".format(checkpoint['epoch']))
    else:
        print("=> No checkpoint found at '{}'".format(modelname))

    Test_isic(testloader, model)
