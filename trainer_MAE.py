import argparse
from models.BiGRU import BiGRU
from models.BiLSTM import BiLSTM
from models.DCNN import DCNN
from dataloader.data_loader import FBGDataset, FBGNoisyDataset
import torch.utils.data as data
import torch
from loss_lib import regularized_nll_loss
import torch.autograd as autograd
import numpy as np
from sklearn import metrics
from utils import TensorboardWriter, get_logger
import os
from loss_lib import MAELoss
import torch.nn as nn


def trainer_nn(model, device, optimizer, train_data, test_data, writer, epochs, early_stop, criterion, clip):
    step = 0
    best_acc = 0
    for epoch in range(epochs):
        print('Epoch: {}'.format(epoch + 1))
        model.train()
        loss_tmp = []
        for i_batch, batch_data in enumerate(train_data):
            data, target = batch_data['data'], batch_data['noisy_labels']
            data = data.float()
            target = target.long()
            data = data.to(device)
            target = target.to(device)
            target = autograd.Variable(target)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip, norm_type=2)
            optimizer.step()
            loss_tmp.append(loss.item())
            writer.update_loss(loss, step, 'step_loss')
            step += 1
        avg_loss, accuracy, precision, recall, f1, kappa, corrects = eval_nll(train_data, model, device, criterion)
        print(
            'Train-Evaluation - loss: {:.6f}  acc: {:.2f}   pre: {:.2f}   recall: {:.2f}  f1: {:.4f}  kappa: {:.4f}   '
            '%({}/{}) ' .format(avg_loss, accuracy, precision * 100, recall * 100, f1, kappa, corrects,
                                len(train_data.dataset)))
        if abs(avg_loss) <= 0.000005:
            print('Loss low')
        writer.update_loss(avg_loss, epoch, 'train_loss')
        writer.update_loss(accuracy, epoch, 'train_accuracy')
        writer.update_loss(precision, epoch, 'train_precision')
        writer.update_loss(recall, epoch, 'train_recall')
        writer.update_loss(f1, epoch, 'train_f1')
        writer.update_loss(kappa, epoch, 'train_kappa')
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            writer.add_histogram(tag, value.data.cpu().numpy(), epoch)
            writer.add_histogram(tag + '/grad', value.grad.data.cpu().numpy(), epoch)
        if accuracy > best_acc:
            model_star = model
            best_acc = accuracy
        avg_loss, accuracy, precision, recall, f1, kappa, corrects = eval_nll(test_data, model, device, criterion)
        print(
            'Test-Evaluation - loss: {:.6f}  acc: {:.2f}   pre: {:.2f}   recall: {:.2f}  f1: {:.4f}  kappa: {:.4f}   '
            '%({}/{}) '.format(avg_loss, accuracy, precision * 100, recall * 100, f1, kappa, corrects,
                               len(test_data.dataset)))
        writer.update_loss(avg_loss, epoch, 'test_loss')
        writer.update_loss(accuracy, epoch, 'test_accuracy')
        writer.update_loss(precision, epoch, 'test_precision')
        writer.update_loss(recall, epoch, 'test_recall')
        writer.update_loss(f1, epoch, 'test_f1')
        writer.update_loss(kappa, epoch, 'test_kappa')
    return model_star


def eval_nll(data_iter, model, device, criterion):
    model.eval()
    corrects, avg_loss = 0, 0
    y_out = np.empty([0])
    y_target = np.empty([0])
    for i_batch, batch_data in enumerate(data_iter):
        feature, target = batch_data['data'], batch_data['label']
        feature = feature.float()
        target = target.long()
        feature, target = feature.to(device), target.to(device)
        output = model(feature)
        loss = criterion(output, target)
        avg_loss += loss.item()
        corrects += (torch.max(output.cpu(), 1)[1].view(target.cpu().size()).data == target.cpu().data).sum()
        y_out_batch = torch.max(output.cpu(), 1)[1].numpy()
        y_out = np.append(y_out, y_out_batch, axis=0)
        y_target = np.append(y_target, target.cpu().numpy(), axis=0)
    size = len(data_iter.dataset)
    avg_loss = loss.item() / size
    accuracy = float(corrects) / size * 100.0
    precision = metrics.precision_score(y_target, y_out, average='macro')
    recall = metrics.recall_score(y_target, y_out, average='macro')
    f1 = metrics.f1_score(y_target, y_out, average='macro')
    kappa = metrics.cohen_kappa_score(y_target, y_out)
    return avg_loss, accuracy, precision, recall, f1, kappa, corrects


if __name__ == '__main__':
    """
    Parse Arguments
    """

    parser = argparse.ArgumentParser(description="Fault Detection of ATC Antenna Beam -- MAE Loss Trainer")

    parser.add_argument("--version", type=int, default=11, help="model version")
    parser.add_argument("--path", type=str, default='./data_utils/', help="root path to dataset (./data_utils/)")
    parser.add_argument('--arch', type=str, default='BiGRU', help='model architectures (BiGRU | TBC)')
    parser.add_argument(
        "--device", type=str, default="cuda", help="device (cuda or cpu)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=256, help="batch size"
    )
    parser.add_argument(
        "--ds", type=int, default=1, help="down-sampling"
    )
    parser.add_argument(
        "--sample_len", type=int, default=1, help="sample length"
    )
    parser.add_argument(
        "--feature_length", type=int, default=1000, help="feature length of a sequence"
    )
    parser.add_argument(
        "--feature_reshape", action="store_true", default=True, help="shape features (True for RNN)"
    )
    parser.add_argument(
        "--epochs_shuffle", action="store_true", default=True, help="shuffle in epochs"
    )
    parser.add_argument(
        "--train_dataset_ratio", type=float, default=0.8, help="training dataset ratio"
    )
    parser.add_argument(
        "--featurizer_hidden_dim", nargs='+', type=int, default=[500, 1000], help="hidden state dimension of featurizer"
    )
    parser.add_argument(
        "--dropout", type=float, default=0.8, help="dropout"
    )
    parser.add_argument(
        "--optimizer", type=str, default='Adam', help="optimizer (Adam or SGD)"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="learning rate"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-8, help="weight decay"
    )
    parser.add_argument(
        "--momentum_value", type=float, default=0.9, help="momentum value, only for SGD"
    )
    parser.add_argument(
        "--epoch", type=int, default=75, help="training epoch"
    )

    parser.add_argument(
        "--early_stop", action="store_true", default=False, help="early stop of training"
    )

    parser.add_argument(
        "--noise_ratio", type=float, default=0.0, help="noise ratio of label noise"
    )

    parser.add_argument(
        "--clip", type=float, default=5, help="gradient clip"
    )

    parser.add_argument(
        "--num_class", type=int, default=3, help="class num of task"
    )


    args = parser.parse_args()
    data_path = args.path + 'data-ds_{:d}-sample_len_{:d}.npy'.format(args.ds, args.sample_len)
    time_path = args.path + 'time-ds_{:d}-sample_len_{:d}.csv'.format(args.ds, args.sample_len)
    label_path = args.path + 'label-ds_{:d}-sample_len_{:d}.csv'.format(args.ds, args.sample_len)
    type_path = args.path + 'type-ds_{:d}-sample_len_{:d}.csv'.format(args.ds, args.sample_len)
    device = torch.device(args.device)
    log_dir = './logs/noisy_training/mae_loss/{}/run_{}/'.format(args.arch, args.version)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    """
    Load Dataset
    """

    dataset = FBGNoisyDataset(data_path=data_path,
                         label_path=label_path,
                         time_path=time_path,
                         sensor_num_path=type_path, feature_length=args.feature_length, reshape=args.feature_reshape,
                              noise_ratio=args.noise_ratio)
    length = dataset.__len__()
    train_size = int(args.train_dataset_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = data.random_split(dataset, [train_size, test_size])
    train_dataloader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=args.epochs_shuffle)
    test_dataloader = data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=args.epochs_shuffle)
    all_dataloader = data.DataLoader(dataset, batch_size=args.batch_size, shuffle=args.epochs_shuffle)

    """
    Load Model
    """
    if args.arch == 'BiGRU':
        model = BiGRU(featurizer_hidden_dim=args.featurizer_hidden_dim, feature_length=args.feature_length,
                    dropout=args.dropout, device=device).to(device)
    if args.arch == 'GRU':
        model = BiGRU(featurizer_hidden_dim=args.featurizer_hidden_dim, feature_length=args.feature_length,
                    dropout=args.dropout, device=device, bidirectional=False).to(device)
    if args.arch == 'BiLSTM':
        model = BiLSTM(featurizer_hidden_dim=args.featurizer_hidden_dim, feature_length=args.feature_length,
                    dropout=args.dropout, device=device).to(device)
    if args.arch == 'LSTM':
        model = BiLSTM(featurizer_hidden_dim=args.featurizer_hidden_dim, feature_length=args.feature_length,
                    dropout=args.dropout, device=device, bidirectional=False).to(device)
    if args.arch == 'DCNN':
        model = DCNN(feature_length=args.feature_length, dropout=args.dropout, device=device).to(device)

    """
    Tensorboard
    """
    writer = TensorboardWriter(log_dir)

    """
    Train Model
    """
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(list(model.parameters()), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(list(model.parameters()), lr=args.learning_rate, weight_decay=args.weight_decay,
                        momentum=args.momentum_value)
    criterion = MAELoss(num_classes=args.num_class)
    model_star = trainer_nn(model=model, device=device, optimizer=optimizer, train_data=train_dataloader,
                            test_data=test_dataloader, writer=writer, epochs=args.epoch,
                            early_stop=args.early_stop, criterion=criterion, clip=args.clip)

    """
    Save Model
    """
    save_path_model = './logs/noisy_training/mae_loss/{}/version_{}.pth'.format(args.arch, args.version)
    torch.save(model_star, save_path_model)
    writer.close()

    """
    Save Logs
    """
    logger = get_logger('./logs/noisy_training/mae_loss/{}/model_log.log'.format(args.arch))
    logger.info('\n')
    args_dict = vars(args)
    for key in args_dict.keys():
        logger.info('{}: {}'.format(key, args_dict[key]))

    """
    python -m tensorboard.main --logdir=./logs/noisy_training/mae_loss/BiGRU/run_2/ --port=6006
    tensorboard --logdir=D:\\Sudao_HE\\ZhuZhou_Project\\logs\\noisy_training\\trunc_loss\\BiGRU\\run_4
    """