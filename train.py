import time
import random
import numpy as np
import pandas as pd
import math
import mxnet as mx
from mxnet import ndarray as nd, gluon, autograd
from mxnet.gluon import loss as gloss
import dgl
from sklearn.model_selection import KFold
from sklearn import metrics


from utils import build_graph, sample
from model import GraphTGI, GraphEncoder, BilinearDecoder

import dgl.function as FN

def Train(directory, epochs, aggregator, embedding_size, layers, dropout, slope, lr, wd, random_seed, ctx):
    dgl.load_backend('mxnet')
    random.seed(random_seed)
    np.random.seed(random_seed)
    mx.random.seed(random_seed)

    g, TF_ids_invmap, tg_ids_invmap, TFSM = build_graph(directory, random_seed=random_seed, ctx=ctx)
    
    samples = sample(directory, random_seed=random_seed)  # 从图结构中获取一些采样样本，并存储到numpy数组 samples中

    samples_df = pd.DataFrame(samples, columns=['TF', 'tg', 'label'])   # 创建samples_df，通过samples传入三列数据
    sample_TF_vertices = [TF_ids_invmap[id_] for id_ in samples[:, 0]]   # 通过TF_ids_invmap将药物的ID映射回图中节点的名称，生成sample_TF_vertices列表
    sample_tg_vertices = [tg_ids_invmap[id_] + TFSM.shape[0] for id_ in samples[:, 1]]  # 将每个样本的tg值进行映射，转换成对应具体的特征值

    print(type(samples))   #sample的类型 <class 'numpy.ndarray'>
    np.savetxt('sample.txt', samples, fmt='%.0f', delimiter=',')
    # print(samples.asnumpy())

    kf = KFold(n_splits=10, shuffle=True, random_state=random_seed)  # 对数据集进行交叉验证划分
    train_index = []  # 列表属性，用来存储训练集的索引信息
    test_index = []

    for train_idx, test_idx in kf.split(samples[:, 2]):   # 生成索引 以将数据拆分为训练集和测试集
        train_index.append(train_idx)
        test_index.append(test_idx)

    print(train_idx)   # train_idx是一个变量，表示当前循环下的训练集索引
    print(test_idx)
    # print(train_index)
    # print(test_index)

    auc_result = []
    acc_result = []
    pre_result = []
    recall_result = []
    f1_result = []

    fprs = []
    tprs = []


    for i in range(len(train_index)):
        print('------------------------------------------------------------------------------------------------------')
        print('Training for Fold ', i + 1)  # 输出当前循环所在的折编号

        samples_df['train'] = 0   # （pd.DataFrame类型）在samples_df 中加入train列
        samples_df['test'] = 0
        
        samples_df['train'].iloc[train_index[i]] = 1  # 把train列对应位置设置为1，标记样本设为训练集
        samples_df['test'].iloc[test_index[i]] = 1    # 把test列对应位置设置为1，标记样本设为测试集

        # 将训练集信息转换成mxnet张量train_tensor 并把他们复制到指定的CPU上
        train_tensor = nd.from_numpy(samples_df['train'].values.astype('int32')).copyto(ctx)
        test_tensor = nd.from_numpy(samples_df['test'].values.astype('int32')).copyto(ctx)

        # 创建一个字典edge_data，有‘train’和‘test’两个键，值分别是train_tensor和test_tensor
        edge_data = {'train': train_tensor,
                     'test': test_tensor}

        # 调用update方法，将新建的字典中的信息更新到g对象的边数据中
        # 从sample_TF_vertices到sample_tg_vertices的所有边对应的边属性字典，通过update方法实现更新
        g.edges[sample_TF_vertices, sample_tg_vertices].data.update(edge_data)
        g.edges[sample_tg_vertices, sample_TF_vertices].data.update(edge_data)


        # g.filter_edges可以筛选出所有有train标记的边，（从训练集节点到目标节点的边），并将结果存在train_eid中
        # lambda，根据train键过滤出指定的边对象
        train_eid = g.filter_edges(lambda edges: edges.data['train']).astype('int64')
        print(len(train_eid))  # 输出训练集样本与目标节点之间所有边的数量
        g_train = g.edge_subgraph(train_eid, preserve_nodes=True)  # 基于train_eid的边ID列表生成一个新的子图对象


        print(len(train_tensor) + 1)     # 改过 原来是len(train_tensor + 1)


        # get the training set
        # rating_train = g_train.edata['rating']
        # 所有被标记为训练集的边，存储在train_eid。
        # 获取训练集边的评分信息
        rating_train = g.edges[train_eid].data['rating']
        src_train, dst_train = g_train.all_edges()  # 获取整个图g中的所有边，并将他们划分为源节点集合src_train，和，目标节点集合dst_train

        # get the testing edge set
        test_eid = g.filter_edges(lambda edges: edges.data['test']).astype('int64')
        src_test, dst_test = g.find_edges(test_eid)
        rating_test = g.edges[test_eid].data['rating']
        src_train = src_train.copyto(ctx)
        src_test = src_test.copyto(ctx)
        dst_train = dst_train.copyto(ctx)
        dst_test = dst_test.copyto(ctx)
        print('## Training edges:', len(train_eid))
        print('## Testing edges:', len(test_eid))

        # print(type(dst_test))
        # print(dst_test)
        # print(type(src_test))
        # print(src_test)

        # print(test_eid)

        # Train the model
        # 创建一个图神经网络模型的对象，由GraphEncoder和BilinearDecoder两个组件构成
        # embedding_size生成节点嵌入的维度 n_layers表示GCN网络的层数 aggregator邻居聚合的方式 slope表示LeakyReLU层中的负斜率参数
        # BilinearDecoder组件负责在节点embeddings的基础上预测边的权重
        model = GraphTGI(GraphEncoder(embedding_size=embedding_size, n_layers=layers, G=g, aggregator=aggregator,
                                    dropout=dropout, slope=slope, ctx=ctx),
                       BilinearDecoder(feature_size=embedding_size))

        # 对模型参数的初始化  model.collect_params()获取所有参数  使用Xavier初始化方法，激活函数为LeakyReLU
        model.collect_params().initialize(init=mx.init.Xavier(magnitude=math.sqrt(2.0)), ctx=ctx)

        # 定义损失函数 先对输入数据激活，在计算二分类交叉熵损失，这个函数将模型的预测结果转换成概率值
        cross_entropy = gloss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=True)
        # 定义优化器
        trainer = gluon.Trainer(model.collect_params(), 'adam', {'learning_rate': lr, 'wd': wd})

        # 模型的训练
        for epoch in range(epochs):
            start = time.time()
            for _ in range(10):
                with mx.autograd.record():
                    # score_train预测的结果  embeddings节点嵌入向量 src_train和dst_train分别表示输入边的源节点和目标节点的ID
                    score_train, embeddings = model(g, src_train, dst_train)
                    loss_train = cross_entropy(score_train, rating_train).mean()  # 计算输出结果和样本标签之间的交叉熵损失，并对所有样本损失求平均
                    loss_train.backward()  # 反向传播 求梯度
                trainer.step(1, ignore_stale_grad=True)  # 根据优化器的设定将梯度传给模型参数，并更新参数

            h_val = model.encoder(g)  # 获得模型的编码器对整个图g进行编码，生成节点的嵌入向量h_val
            # 通过解码器 根据源节点和目标节点的嵌入向量h_val[src_test]和h_val[dst_test] 计算边权重得分score_val
            score_val = model.decoder(h_val[src_test], h_val[dst_test])
            # 计算交叉熵损失
            loss_val = cross_entropy(score_val, rating_test).mean()

            # metrics.roc_auc_score 用于计算指定分类器在指定数据集上的ROC AUC值
            # np.squeeze(rating_train.asnumpy() 表示训练集的样本标签 通常是0或1
            # np.squeeze(score_train.asnumpy()) 训练集的输出得分，其值代表分类器预测为1的概率
            train_auc = metrics.roc_auc_score(np.squeeze(rating_train.asnumpy()), np.squeeze(score_train.asnumpy()))
            val_auc = metrics.roc_auc_score(np.squeeze(rating_test.asnumpy()), np.squeeze(score_val.asnumpy()))

            results_val = [0 if j < 0.5 else 1 for j in np.squeeze(score_val.asnumpy())]
            accuracy_val = metrics.accuracy_score(rating_test.asnumpy(), results_val)  # 计算准确率
            precision_val = metrics.precision_score(rating_test.asnumpy(), results_val)  # 计算查准率
            recall_val = metrics.recall_score(rating_test.asnumpy(), results_val)  # 计算查全率
            f1_val = metrics.f1_score(rating_test.asnumpy(), results_val)  # 计算f1值

            end = time.time()

            print('Epoch:', epoch + 1, 'Train Loss: %.4f' % loss_train.asscalar(),
                  'Val Loss: %.4f' % loss_val.asscalar(),
                  'Acc: %.4f' % accuracy_val, 'Pre: %.4f' % precision_val, 'Recall: %.4f' % recall_val,
                  'F1: %.4f' % f1_val, 'Train AUC: %.4f' % train_auc, 'Val AUC: %.4f' % val_auc,
                  'Time: %.2f' % (end - start))
        

        h_test = model.encoder(g)  # 对测试集中的图g进行编码
        score_test = model.decoder(h_test[src_test], h_test[dst_test])  # 对测试集图g进行解码，得到输出得分score_test

        print(type(score_test))
        score_test1 = score_test.asnumpy()  # 将score_test转换成numpy数组并保存到score_test1
        np.savetxt('score_test1', score_test1, fmt='%.0f', delimiter=',')

        # metrics.roc_curve用于计算ROC曲线的FPR、TPR和阈值thresholds
        # np.squeeze(rating_test.asnumpy()) 为测试集的样本标签
        # np.squeeze(score_test.asnumpy()) 为测试集的输出得分
        fpr, tpr, thresholds = metrics.roc_curve(np.squeeze(rating_test.asnumpy()), np.squeeze(score_test.asnumpy()))
        test_auc = metrics.auc(fpr, tpr)  # 计算ROC曲线下的面积

        np.savetxt('array1.txt', np.squeeze(score_test.asnumpy()))   #输出得分


        results_test = [0 if j < 0.5 else 1 for j in np.squeeze(score_test.asnumpy())]
        accuracy_test = metrics.accuracy_score(rating_test.asnumpy(), results_test)
        precision_test = metrics.precision_score(rating_test.asnumpy(), results_test)
        recall_test = metrics.recall_score(rating_test.asnumpy(), results_test)
        f1_test = metrics.f1_score(rating_test.asnumpy(), results_test)

        print('Fold:', i + 1, 'Test Acc: %.4f' % accuracy_test, 'Test Pre: %.4f' % precision_test,
              'Test Recall: %.4f' % recall_test, 'Test F1: %.4f' % f1_test, 'Test AUC: %.4f' % test_auc)

        auc_result.append(test_auc)
        acc_result.append(accuracy_test)
        pre_result.append(precision_test)
        recall_result.append(recall_test)
        f1_result.append(f1_test)

        fprs.append(fpr)  #
        tprs.append(tpr)
    
    print('## Training Finished !')
    print('----------------------------------------------------------------------------------------------------------')

    return auc_result, acc_result, pre_result, recall_result, f1_result, fprs, tprs,embeddings
