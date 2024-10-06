import argparse
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from torchvision import transforms

from models.LSTMAE import LSTMAE
from models.LSTM_PRED import LSTMAEPRED
from train_utils import train_model, eval_model

parser = argparse.ArgumentParser(description='SFOC code')
parser.add_argument('--batch-size', type=int, default=20, metavar='N',
                    help='input batch size for training (default: 20)')  # --批的大小--
parser.add_argument('--epochs', type=int, default=256, metavar='N', help='number of epochs to train')  # --训练的周期个数--
parser.add_argument('--optim', default='Adam', type=str, help='Optimizer to use')
parser.add_argument('--hidden-size', type=int, default=10, metavar='N', help='LSTM hidden state size')  # --隐层节点数量--
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR', help='learning rate')  # --学习率--
parser.add_argument('--input-size', type=int, default=20, metavar='N', help='input size')  # --输入特征--
parser.add_argument('--dropout', type=float, default=0.0, metavar='D', help='dropout ratio')
parser.add_argument('--wd', type=float, default=0, metavar='WD', help='weight decay')
parser.add_argument('--grad-clipping', type=float, default=1, metavar='GC', help='gradient clipping value')
parser.add_argument('--log-interval', type=int, default=1, metavar='N', help='how many batch iteration to log status')
parser.add_argument('--model-dir', default='trained_models', help='directory of model for saving checkpoint')
parser.add_argument('--model-type', default='LSTMAE_PRED', help='type of model to use: LSTMAE or LSTMAE_PRED for '
                                                           'prediction')
parser.add_argument('--seq-len', default=12, help='sequence full length') # --输入序列的长度--
parser.add_argument('--cross-val', type=int, default=4, help='number of cross validation experiments to run')
parser.add_argument('--data-path', type=str, default='data/Augest-Shanghai-Dalian.csv',
                    help='Path to the SFOC csv data') # --数据集位置--

args = parser.parse_args(args=[])

kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    trained_model, val_loss, test_X, test_Y, foc_scaling_metadata = run_experiment()

    # Save model
    torch.save(trained_model.state_dict(),
               os.path.join(args.model_dir, f'SFOC model={args.model_type}_hs={args.hidden_size}_bs={args.batch_size}'
                                            f'_epochs={args.epochs}_clip={args.grad_clipping}.pt'))
    # Generate final test iterator and plot reconstructed graphs
    test_iter = torch.utils.data.DataLoader(SFOC_dataset(test_X, test_Y, foc_scaling_metadata),
                                            batch_size=args.batch_size, shuffle=False)
    plot_images(trained_model, test_iter, foc_scaling_metadata)

def run_experiment():
    """
    Main function to experiment data prepare on the SFOC reconstruction/prediction
    :return:
    """
    # Load data into DataFrame
    df = pd.read_csv(args.data_path)

    # data prepare
    df = df.iloc[:, 2:] #--去掉日期和时间项
    df = df.values[::3, :] #--间隔采样，将10秒-->30秒

    # Normalize data to be in [0,1] with mean 0.5
    df, foc_scaling_metadata = normalize_data(df)  #df已经是numpy.ndarray

    # Generate data loaders
    train_Feature, val_Feature, test_Feature, train_Label, val_Label, test_Label = get_train_test_data(df)

    # Get model, optimizer and loss
    model = get_model()
    optimizer = getattr(torch.optim, args.optim)(params=model.parameters(), lr=args.lr, weight_decay=args.wd)
    criterion = rec_pred_loss if args.model_type == 'LSTMAE_PRED' else torch.nn.MSELoss(reduction='sum')

    # Getnerate batch data loaders
    batch_size = args.batch_size
    train_iter = torch.utils.data.DataLoader(dataset=SFOC_dataset(train_Feature, train_Label, transform=transforms.ToTensor()),
                              batch_size=batch_size,  drop_last=True, shuffle=True)
    #test_iter = torch.utils.data.DataLoader(dataset=SFOC_dataset(test_Feature, test_Label),
    #                                        batch_size=batch_size,  drop_last=True, shuffle=True)
    val_iter = torch.utils.data.DataLoader(dataset=SFOC_dataset(val_Feature, val_Label),
                                            batch_size=batch_size, drop_last=True, shuffle=True)

    # Train and Validation loop 分别为预测损失,自编码损失,验证集损失
    pred_loss_lst, train_loss_lst, val_loss = run_training_loop(criterion, train_iter, val_iter, model, optimizer)

    # Plot training losses and prediction loss graphs
    plot_train_pred_losses(train_loss_lst, pred_loss_lst, description='SFOC val')

    return model, val_loss, test_Feature, test_Label, foc_scaling_metadata

def normalize_data(df):
    """
    Function to normalize the data in [0,1] range with mean 0.5
    :param df: raw loaded data
    :return: data after normalization
    """
    # Normalize and center each sequence around 0.5
    foc_scaling_metadata = {} #SFOC scaling and mean
    Foc_max, Foc_min = df[:, 0].max(), df[:, 0].min()
    scaler = preprocessing.MinMaxScaler().fit(df)
    df = scaler.transform(df)
    foc_scaling_metadata[0] = {'Foc_max': Foc_max, 'Foc_min': Foc_min}

    return df, foc_scaling_metadata

def get_train_test_data(df, split_ratio=0.8):
    """
    Function to create the initial Train & Test split. The train will be further split in the cross validation.
    The test will be used for the final evaluation.
    In addition, the function calculate the normalization and time series and random samples.
    :param split_ratio: split ratio to create train/test data
    :return: Train and Test dataFrames
    """
    # Get time series with random
    seq_len = args.seq_len
    # 对数据集进行滑动窗口分割，窗口大小为1
    X = []
    Y = []
    for i in range(df.shape[0] - seq_len):
        X.append(np.array(df[i:(i + seq_len), 2:], dtype=np.float32)) #去掉前两项与SFOC的主机FOC
        Y.append(np.array(df[i+seq_len, 0], dtype=np.float32))
    total_len = df.shape[0]
    # Split train and test SFOC time series,shuffle=True 打乱顺序, 训练集、验证集和测试集采用7:1:2
    X_train_val, X_test, Y_train_val, Y_test = train_test_split(X, Y, train_size=split_ratio, random_state = 0, shuffle= True)
    # train dataset split to train and val
    X_train, X_val, Y_train, Y_val = train_test_split(X_train_val, Y_train_val, train_size=0.875, random_state = 0, shuffle = True)

    # 修改test集
    X_test, Y_test = X[-int(0.2 * total_len):], Y[-int(0.2 * total_len):]


    return X_train, X_val, X_test, Y_train, Y_val, Y_test


class SFOC_dataset(torch.utils.data.Dataset):
    def __init__(self, x, y, Sfoc_scaling_metadata =None, transform=None):
        self.x = x
        self.y = y
        self.sfoc_scaling_metadata = Sfoc_scaling_metadata
        self.transform = transform

    def __getitem__(self, index):
        x_t = self.x[index]
        y_t = self.y[index]
        #         if self.transform !=None:
        #             return self.transform(x_t),y_t
        return x_t, y_t

    def __len__(self):
        return len(self.x)

def run_training_loop(criterion, train_iter, val_iter, model, optimizer):
    """
    Training loop - including training and validation
    :param criterion: loss function
    :param train_iter: train data loader for current fold
    :param val_iter: validation data loader for current fold
    :param model: model to train
    :param optimizer: optimizer to use
    :return:
    """

    # Run train & validation evaluation for epoch
    train_loss_lst, pred_loss_lst = [], []
    for epoch in range(args.epochs):
        # Train loop
        train_loss, _, pred_loss = train_model(criterion, epoch, model, args.model_type,
                                               optimizer, train_iter, args.batch_size,
                                               args.grad_clipping, args.log_interval)
        train_loss_lst.append(train_loss)
        pred_loss_lst.append(pred_loss)
        val_loss, val_acc = eval_model(criterion, model, args.model_type, val_iter, mode='Val')
    return pred_loss_lst, train_loss_lst, val_loss

def get_model():
    """
    Generate the requested model instance
    :return: Pytorch model instance
    """
    # Generate model
    if args.model_type == 'LSTMAE':
        print('Creating LSTM AE model')
        model = LSTMAE(input_size=args.input_size, hidden_size=args.hidden_size, dropout_ratio=args.dropout,
                       seq_len=args.seq_len)
    elif args.model_type == 'LSTMAE_PRED':
        print('Creating LSTM AE with Predictor')
        model = LSTMAEPRED(input_size=args.input_size, hidden_size=args.hidden_size, dropout_ratio=args.dropout,
                           seq_len=args.seq_len)
    else:
        raise NotImplementedError(f'Selected model type is not implemented: {args.model_type}')
    model = model.to(device)
    return model

def rec_pred_loss(data_rec, data, out_preds, real_pred):
    """
    Function that calculated both reconstruction and prediction loss
    :param data_rec: reconstructed data instance
    :param data: original data instance
    :param out_preds: predicted next price
    :param real_pred: actual next price
    :return: the reconstruction loss and prediction loss
    """
    mse_rec = torch.nn.MSELoss(reduction='sum')(data_rec, data)
    mse_pred = torch.nn.MSELoss(reduction='sum')(out_preds, real_pred)

    return mse_rec, mse_pred

def plot_train_pred_losses(train_loss_lst, prediction_loss_lst, description):
    """
    Plots the training loss graph vs. epoch and the prediction loss grpah vs. epoch.
    :param train_loss_lst: list of all train loss recorded from the training process
    :param prediction_loss_lst: list of all prediction losses recorded from the training process
    :param description: additional description to add to the plots
    :return:
    """
    epochs = [t for t in range(len(train_loss_lst))]

    plt.figure()
    plt.plot(epochs, prediction_loss_lst, color='r', label='Train prediction loss')
    plt.xticks([i for i in range(0, len(epochs) + 1, int(len(epochs) / 10))])
    plt.xlabel('Epochs')
    plt.ylabel(f'{description} Prediction loss value')
    plt.legend()
    plt.title(f'{description} Prediction loss vs. epochs')
    plt.savefig(f'{description}_prediction_graphs.png')
    plt.show()

    plt.figure()
    plt.plot(epochs, train_loss_lst, color='r', label='Train loss')
    plt.xticks([i for i in range(0, len(epochs) + 1, int(len(epochs) / 10))])
    plt.xlabel('Epochs')
    plt.ylabel(f'{description} Train loss value')
    plt.legend()
    plt.title(f'{description} Train loss vs. epochs')
    plt.savefig(f'{description}_train_loss_graphs.png')
    plt.show()


def plot_images(model, test_iter, foc_scaling_metadata):
    """
    Plots the original vs. reconstructed graphs
    :param model: trained model to reconstruct stock price graphs
    :param test_iter: test data loader   :return:
    """
    model.eval()
    # Inverse transform prices
    Foc_max = foc_scaling_metadata[0]['Foc_max']
    Foc_min = foc_scaling_metadata[0]['Foc_min']

    test_labels = []
    test_preds = []

    for index, (x, label) in enumerate(test_iter):
        x = x.squeeze(1)  # batch_size,seq_len,input_size
        rec_data = model(x.to(device))
        if len(rec_data) > 1:
            rec_data = rec_data[1]
        test_preds.extend((rec_data.squeeze() * (Foc_max - Foc_min) + Foc_min).tolist())
        test_labels.extend((label.squeeze() * (Foc_max - Foc_min) + Foc_min).tolist())

    # orig = data[i].squeeze()
    plt.figure()
    # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    # plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=101))
    plt.grid()
    plt.plot(test_labels, color='g', label='Ground Truth signal')
    plt.plot(test_preds, color='r', label='Prediction signal')
    plt.xlabel('TimeSeries')
    plt.ylabel('Signal Value')
    plt.legend()
    plt.title(f'Original and Rec for test example')
    plt.savefig(f'SFOC orig vs pred')
    plt.show()


if __name__ == '__main__':
    main()