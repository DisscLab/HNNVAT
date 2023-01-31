import time
import pandas as pd
import torch
import torch.optim as optim
from sklearn import metrics
import numpy as np
import sys

sys.path.insert(0, 'lib/')
import lib.utilsdata
# from pycm import *
import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F


# from lib.vat import VATLoss

def calculation(score_normed, data_one_hot, pred_test, test_labels, method='GCN'):  
    test_labels = np.array(test_labels)
    test_acc = metrics.accuracy_score(pred_test, test_labels)
    test_f1_macro = metrics.f1_score(pred_test, test_labels, average='macro')
    test_f1_micro = metrics.f1_score(pred_test, test_labels, average='micro')
    precision = metrics.precision_score(pred_test, test_labels, average='micro')
    recall = metrics.recall_score(pred_test, test_labels, average='micro')

    fpr, tpr, thresholds = metrics.roc_curve(pred_test, test_labels, pos_label=2)
    auc = metrics.auc(fpr, tpr)

    # print('test_acc','f1_test_macro','f1_test_micro','Testprecision','Testrecall','Testauc')
    # print( test_acc, test_f1_macro, test_f1_micro, precision,recall,auc )
    print('---------------------------------------------------------------------------------------------------------')

    print('test_acc= %.4f, f1_test_macro= %.4f, f1_test_micro= %.4f, Testprecision= %.4f, Testrecall= %.4f, auc= %.4f' %
          (test_acc, test_f1_macro, test_f1_micro, precision, recall, auc))

    # cm = ConfusionMatrix(actual_vector=test_labels, predict_vector=pred_test)
    print('---------------------------------------------------------------------------------------------------------')
    # print(cm)
    # print(cm.overall_stat)
    # cm.plot()


def weight_init(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight) 
        if m.bias is not None:
            m.bias.data.fill_(0.0) 

def test_model(net, loader, L, args):
    t_start_test = time.time()

    net.eval()
    test_acc = 0
    count = 0
    confusionGCN = np.zeros([args.nclass, args.nclass])  
    predictions = pd.DataFrame()  
    y_true = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        out_gae, out_hidden, pred = net(batch_x, args.dropout, L)
        test_acc += lib.utilsdata.accuracy(pred, batch_y).item() * len(batch_y)
        count += 1
        y_true = batch_y.detach().cpu().numpy()  
        y_predProbs = pred.detach().cpu().numpy()  
    predictions = pd.DataFrame(y_predProbs) 
    for i in range(len(y_true)):  
        confusionGCN[y_true[i], np.argmax(y_predProbs[i, :])] += 1
    t_total_test = time.time() - t_start_test
    preds_labels = np.argmax(np.asarray(predictions), 1)  

    test_acc = test_acc / len(loader.dataset)

    return test_acc, confusionGCN, predictions, preds_labels, y_true, t_total_test


def train_model(useModel, train_loader, val_loader, L, args):
    D_g = args.num_gene  
    CL1_F = 1  
    CL1_K = 5  
    FC1_F = 32 
    FC2_F = 0  
    NN_FC1 = 256  
    NN_FC2 = 32  
    out_dim = args.nclass
    net_parameters = [D_g, CL1_F, CL1_K, FC1_F, FC2_F, NN_FC1, NN_FC2, out_dim]
    # learning parameters
    dropout_value = 0.3
    l2_regularization = 5e-4  
    batch_size = args.batchsize  
    num_epochs = args.epochs
    nb_iter = int(num_epochs * args.train_size) 
    print('num_epochs=', num_epochs, ', train_size=', args.train_size, ', nb_iter=', nb_iter)
    global_lr = args.lr
    global_step = 0
    decay = 0.95
    decay_steps = args.train_size

    net = useModel(net_parameters)  
    net.apply(weight_init)  
    if torch.cuda.is_available():
        net.cuda()
    print(net)  

    optimizer = optim.SGD(net.parameters(), momentum=0.9, lr=args.lr)  # SGD
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.train()
    losses_train = []
    acc_train = [] 
    t_total_train = time.time() 

    def adjust_learning_rate(optimizer, epoch, lr):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = lr * pow(decay, float(global_step // decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    for epoch in range(num_epochs):

        cur_lr = adjust_learning_rate(optimizer, epoch, args.lr)  
        t_start = time.time()
        epoch_loss = 0.0
        epoch_acc = 0.0
        count = 0
        for i, (batch_x, batch_y) in enumerate(train_loader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()

            out_gae, out_hidden, output = net(batch_x, dropout_value, L)

            loss_batch = net.loss(out_gae, batch_x, output, batch_y, l2_regularization)
            vat_lr = 0.1
            if vat_lr > 0:
                # batch_x = batch_x.to(device)
                # x 64*1000  y:64
                vat_loss = VATLoss(xi=10.0, eps=1.0, ip=1) 
                lds = vat_loss(net, batch_x, L, output)
            loss_batch += lds * vat_lr

            acc_batch = lib.utilsdata.accuracy(output, batch_y).item()

            loss_batch.backward()
            optimizer.step()

            count += 1
            epoch_loss += loss_batch.item() 
            epoch_acc += acc_batch
            global_step += args.batchsize
            if count % 1000 == 0:  # print every x mini-batches
                print('epoch= %d, i= %4d, loss(batch)= %.4f, accuray(batch)= %.2f' % (
                epoch + 1, count, loss_batch.item(), acc_batch))

        epoch_loss /= count
        epoch_acc /= count

        losses_train.append(epoch_loss)  
        acc_train.append(epoch_acc) 
        t_stop = time.time() - t_start

        if (epoch + 1) % 10 == 0 and (epoch + 1) != 0:  
            with torch.no_grad():
                val_acc = 0
                count = 0
                for b_x, b_y in val_loader:
                    b_x, b_y = b_x.to(device), b_y.to(device)
                    _, _, val_pred = net(b_x, args.dropout, L)
                    val_acc += lib.utilsdata.accuracy(val_pred, b_y).item() * len(b_y)
                    count += 1
                val_acc = val_acc / len(val_loader.dataset)

            print('epoch= %d, loss(train)= %.3f, accuracy(train)= %.3f, time= %.3f, lr= %.5f' %
                  (epoch + 1, epoch_loss, epoch_acc, t_stop, cur_lr))
            print('accuracy(val)= ', val_acc)
        else:
            print('epoch= %d, loss(train)= %.3f, accuracy(train)= %.3f, time= %.3f, lr= %.5f' %
                  (epoch + 1, epoch_loss, epoch_acc, t_stop, cur_lr))

    t_total_train = time.time() - t_total_train

    return net, t_total_train


@contextlib.contextmanager
def _disable_tracking_bn_stats(model):
    def switch_attr(m):
        if hasattr(m, 'track_running_stats'):
            m.track_running_stats ^= True

    model.apply(switch_attr)
    yield
    model.apply(switch_attr)


def _l2_normalize(d):
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
    return d


class VATLoss(nn.Module):

    def __init__(self, xi=10.0, eps=1.0, ip=1):
        """VAT loss
        :param xi: hyperparameter of VAT (default: 10.0)
        :param eps: hyperparameter of VAT (default: 1.0)
        :param ip: iteration times of computing adv noise (default: 1)
        """
        super(VATLoss, self).__init__()
        self.xi = xi
        self.eps = eps
        self.ip = ip

    def forward(self, model, x, L, pred):
        with torch.no_grad():
            dropout_value = 0.3
            # pred = pred
            x_decode_gae, x_hidden_gae, pred = model(x, dropout_value, L)

        d = torch.rand(x.shape).sub(0.5).to(x.device)  
        d = _l2_normalize(d)  
        with _disable_tracking_bn_stats(model):
            for _ in range(self.ip):
                d.requires_grad_()
                x_decode_gae, x_hidden_gae, pred_hat = model(x + self.xi * d, dropout_value, L)
                logp_hat = torch.log(pred_hat)

                adv_distance = F.kl_div(logp_hat, pred, reduction='batchmean')
                adv_distance.backward()
                d = _l2_normalize(d.grad)
                model.zero_grad()

            r_adv = d * self.eps
            x_decode_gae, x_hidden_gae, pred_hat = model(x + r_adv, dropout_value, L)
            logp_hat = torch.log(pred_hat)
            lds = F.kl_div(logp_hat, pred, reduction='batchmean')

        return lds