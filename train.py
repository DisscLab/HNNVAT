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

def calculation(score_normed, data_one_hot, pred_test, test_labels, method='GCN'):  # 第一个是预测lable 第二个是truelable
    test_labels = np.array(test_labels)
    # 5个指标
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

    print('test_acc= %.3f, f1_test_macro= %.4f, f1_test_micro= %.4f, Testprecision= %.4f, Testrecall= %.4f, auc= %.4f' %
          (test_acc, test_f1_macro, test_f1_micro, precision, recall, auc))

    # cm = ConfusionMatrix(actual_vector=test_labels, predict_vector=pred_test)
    print('---------------------------------------------------------------------------------------------------------')
    # print(cm)
    # print(cm.overall_stat)
    # cm.plot()


def weight_init(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)  # w服从均匀分布
        if m.bias is not None:
            m.bias.data.fill_(0.0)  # bias初始值为0


def test_model(net, loader, L, args):
    t_start_test = time.time()

    net.eval()
    test_acc = 0
    count = 0
    confusionGCN = np.zeros([args.nclass, args.nclass])  # (13,13) 全0,预测对的标1
    predictions = pd.DataFrame()  # 创建空dataframe，存储预测值
    y_true = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        # 模型预测
        out_gae, out_hidden, pred = net(batch_x, args.dropout, L)
        test_acc += lib.utilsdata.accuracy(pred, batch_y).item() * len(batch_y)
        count += 1
        y_true = batch_y.detach().cpu().numpy()  # 真实值 # (189,)
        y_predProbs = pred.detach().cpu().numpy()  # 预测值
        # 生成的pred （189，13） 是负值 是13个选最大的
    predictions = pd.DataFrame(y_predProbs)  # (189,13)
    for i in range(len(y_true)):  # rgmax返回最大值的索引预测
        confusionGCN[y_true[i], np.argmax(y_predProbs[i, :])] += 1
        # 比如对第一个cell 输入 4和对应的大小13的array
        # confusionGCN (13,13) 的全0矩阵,有预测对的结果矩阵对应位置 + 1 ，类似一个评估矩阵，正确和错误结果都展示
    t_total_test = time.time() - t_start_test
    preds_labels = np.argmax(np.asarray(predictions), 1)  # 预测的lable(189,)

    test_acc = test_acc / len(loader.dataset)

    return test_acc, confusionGCN, predictions, preds_labels, y_true, t_total_test


def train_model(useModel, train_loader, val_loader, L, args):
    D_g = args.num_gene  # 基因数1000
    CL1_F = 1  # 卷积channel
    CL1_K = 5  # 卷积核
    FC1_F = 32  # gcn的FC
    FC2_F = 0  # ?
    NN_FC1 = 64  # NN第1层神经元个数
    NN_FC2 = 32  # NN第2层神经元个数
    out_dim = args.nclass
    net_parameters = [D_g, CL1_F, CL1_K, FC1_F, FC2_F, NN_FC1, NN_FC2, out_dim]
    # learning parameters
    dropout_value = 0.3
    l2_regularization = 5e-4  # 5e-4 = 0.0005  5×10的-4次方
    batch_size = args.batchsize  # 64
    num_epochs = args.epochs
    # 迭代次数 没用到？
    nb_iter = int(num_epochs * args.train_size) // batch_size  # 100*1508/64=2356
    print('num_epochs=', num_epochs, ', train_size=', args.train_size, ', nb_iter=', nb_iter)
    # Optimizer参数
    global_lr = args.lr
    global_step = 0
    decay = 0.95
    decay_steps = args.train_size

    # 将参数传到网络
    net = useModel(net_parameters)  # 调用layermodel
    net.apply(weight_init)  # net.apply()可将weight_init应用于每一个子模块以及父模块
    if torch.cuda.is_available():
        net.cuda()
    print(net)  # 打印

    # optimizer = optim.Adam(net.parameters(),lr= args.lr, weight_decay=5e-4)
    optimizer = optim.SGD(net.parameters(), momentum=0.9, lr=args.lr)  # SGD
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    ## 开始Train
    net.train()
    losses_train = []
    acc_train = []  # 记录指标
    t_total_train = time.time()  # 返回当前时间的时间戳

    # 调整学习率
    def adjust_learning_rate(optimizer, epoch, lr):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        # 变化学习率
        # lr = args.lr * (0.1 ** (epoch // 30))
        lr = lr * pow(decay, float(global_step // decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    for epoch in range(num_epochs):

        cur_lr = adjust_learning_rate(optimizer, epoch, args.lr)  # 优化器里用
        t_start = time.time()
        # 每次epoch指标初始化
        epoch_loss = 0.0
        epoch_acc = 0.0
        count = 0
        # i：0 对数据batch处理，就是把两个数据中cell1508换为64 (64,1000) (64,)
        for i, (batch_x, batch_y) in enumerate(train_loader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()

            # 训练网络 输入batch的表达值矩阵
            out_gae, out_hidden, output = net(batch_x, dropout_value, L)
            # output：pred test是189*13  train不是： 64*13 所以不能直接可视化

            # 输出重构的adj、隐层x(conv结果)、模型输出
            loss_batch = net.loss(out_gae, batch_x, output, batch_y, l2_regularization)
            # 生成最终损失

            # VAT 多消耗三倍的时间

#             vat_lr = 1
#             if vat_lr > 0:
#                 # batch_x = batch_x.to(device)
#                 # x 64*1000  y:64
#                 vat_loss = VATLoss(xi=10.0, eps=1.0, ip=1) # 先迭代ip次 xi是迭代时噪声大小，eps是最终噪声大小
#                 lds = vat_loss(net, batch_x, L, output)
#                 lds = lds / batch_size
#             loss_batch += lds * vat_lr # 损失比率lr

            vat_lr = 0.1
            if vat_lr > 0:
                # batch_x = batch_x.to(device)
                # x 64*1000  y:64
                vat_loss = VATLoss(xi=10.0, eps=1.0, ip=1) # 先迭代ip次 xi是迭代时噪声大小，eps是最终噪声大小
                lds = vat_loss(net, batch_x, L, output)
            loss_batch += lds * vat_lr # 损失比率lr

            acc_batch = lib.utilsdata.accuracy(output, batch_y).item()
            # 生成acc (刚开始train很低)

            loss_batch.backward()
            optimizer.step()

            count += 1
            epoch_loss += loss_batch.item()  # + 别的append之后求平均，这个 +了之后除去 count
            epoch_acc += acc_batch
            global_step += args.batchsize
            # print
            if count % 1000 == 0:  # print every x mini-batches
                print('epoch= %d, i= %4d, loss(batch)= %.4f, accuray(batch)= %.2f' % (
                epoch + 1, count, loss_batch.item(), acc_batch))

        epoch_loss /= count
        epoch_acc /= count

        losses_train.append(epoch_loss)  # 最终loss
        acc_train.append(epoch_acc)  # 最终acc
        t_stop = time.time() - t_start

        # val 不用等没模型跑完也可以看到效果，防止过拟合
        if (epoch + 1) % 10 == 0 and (epoch + 1) != 0:  # 每10个epoch 用一下val
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
        # 直接把第一次的output当做pred传入
        with torch.no_grad():
            dropout_value = 0.3
            # pred = pred
            x_decode_gae, x_hidden_gae, pred = model(x, dropout_value, L)

        # prepare random unit tensor 噪声d 随机x的形状并正则化
        d = torch.rand(x.shape).sub(0.5).to(x.device)  # d：64*1000 跟x一样 sub查询替换
        d = _l2_normalize(d)  # 正则化

        with _disable_tracking_bn_stats(model):
            # calc adversarial direction 计算扰动方向，循环一次（越多越好，但没性能） model输入处加上扰动d*参数xi倍
            # 训练一次扰动为*10 下游为*1
            for _ in range(self.ip):
                d.requires_grad_()
                # 对抗输出的pred，有softmax 只求log
                x_decode_gae, x_hidden_gae, pred_hat = model(x + self.xi * d, dropout_value, L)
                logp_hat = torch.log(pred_hat)

                # 两次对比学习 DL散度Loss：adv_distance
                adv_distance = F.kl_div(logp_hat, pred, reduction='batchmean')
                adv_distance.backward()
                d = _l2_normalize(d.grad)
                model.zero_grad()

            # calc LDS 输出的d再乘以eps 重新输入模型,再进行一次最终对比 得到损失
            r_adv = d * self.eps # 小r adv
            x_decode_gae, x_hidden_gae, pred_hat = model(x + r_adv, dropout_value, L)
            logp_hat = torch.log(pred_hat)

            lds = F.kl_div(logp_hat, pred, reduction='batchmean')

        return lds