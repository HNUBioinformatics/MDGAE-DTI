import mxnet as mx
from mxnet import ndarray as nd
from mxnet.gluon import nn
import dgl
from mxnet.gluon.nn import activations
from mxnet.ndarray.gen_op import Activation
import numpy as np
from dgl.nn.pytorch import GATConv
from attention import cbam_block

class GraphTGI(nn.Block):
    def __init__(self, encoder, decoder):
        super(GraphTGI, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, G, TF, tg):
        h = self.encoder(G)     # 将图G中每个节点标签映射成低维度的向量h
        
        h_TF = h[TF]      # TF药物节点集合，从h中得到与其对应的节点特征向量h_TF
        h_tg = h[tg]      # tg靶标节点集合，从h中得到与其对应的节点特征向量h_tg

        
        return self.decoder(h_TF, h_tg),G.ndata['h']   #通过decoder模块，将上面两部分信息进行融合并得到预测结果，同时返回所有节点的表示


class GraphEncoder(nn.Block):
    def __init__(self, embedding_size, n_layers, G, aggregator, dropout, slope, ctx):
        super(GraphEncoder, self).__init__()

        self.G = G

        # self.TF_nodes表示筛选出图G中type属性为1(即代表TF节点)的所有节点;
        # 并将其数据类型转化为np.int64，最后将它们存储到变量self.TF_nodes中。
        # 这样做的目的是为了在后续的计算过程中，能够更方便地从节点特征矩阵中获取出药物节点对应的特征向量 和靶基因节点对应的特征向量
        # 从而进一步地进行图卷积运算和预测操作
        self.TF_nodes = G.filter_nodes(lambda nodes: nodes.data['type'] == 1).astype(np.int64).copyto(ctx)
        self.tg_nodes = G.filter_nodes(lambda nodes: nodes.data['type'] == 0).astype(np.int64).copyto(ctx)

        self.layers = nn.Sequential()
        
        in_feats = embedding_size

        # 添加两层GATconv图卷积层
        # 对输入的图G进行信息传递和特征提取，以得到节点的嵌入向量表示
        self.layers.add(GATConv(embedding_size, embedding_size, 2, feat_drop=dropout, attn_drop=0.5, negative_slope=0.5, residual=True))#jiale collect_params
        self.layers.add(GATConv(embedding_size, embedding_size, 2, feat_drop=dropout, attn_drop=0.5, negative_slope=0.5, residual=True))
        cbam_block(embedding_size)

        # 是一个神经网络模型中的两个层，分别叫做TFEmbedding和tgEmbedding，
        # 用来将输入数据（比如文本或序列数据）中的词汇转换成向量形式。
        # 这些向量在后续的神经网络中可以被进一步处理和使用。
        self.TF_emb = TFEmbedding(embedding_size, dropout)
        self.tg_emb = tgEmbedding(embedding_size, dropout)

    # 接收一个名为G的图数据结构作为输入，并输出最终的节点嵌入向量
    def forward(self, G):
        # Generate embedding on disease nodes and mirna nodesd
        # 通过assert语句确保输入的G图与实例化模型时使用的图具有相同的节点数
        assert G.number_of_nodes() == self.G.number_of_nodes()

        # 使用DGL库中的apply_nodes方法，对指定节点集合执行一个函数，并将函数的返回值更新到节点的特征向量中
        # lambda函数将TF_emb方法应用于图G中的TF_nodes节点集合。该方法将节点的输入特征向量映射到低维的嵌入向量空间。
        # 将其应用于节点集合TF_nodes之后，得到了这些节点的嵌入向量，并将这些嵌入向量保存在节点特征矩阵中的'h'列中
        # apply_nodes()函数中的第二个参数是一个列表，表示需要进行操作的节点类型
        # 对输入图中self.TF_nodes和self.tg_nodes对应的节点，使用模型中定义的TFEmbedding
        # 和tgEmbedding层对象进行嵌入向量的生成和处理，并更新输入图的节点数据中的'h'键对应的值为生成的节点嵌入向量。
        G.apply_nodes(lambda nodes: {'h': self.TF_emb(nodes.data)}, self.TF_nodes)  # 节点嵌入
        G.apply_nodes(lambda nodes: {'h': self.tg_emb(nodes.data)}, self.tg_nodes)

        # 循环遍历模型中所有的层，调用每一层的forward()方法，
        # 对输入G图的节点嵌入向量进行进一步处理和更新
        for layer in self.layers:
            layer(G, G.ndata['h'])
        cbam_block(G.ndata['h'])
        # 返回更新后的节点嵌入向量，该向量表示了输入图中的每个节点在模型学习过程中的表征。
        return G.ndata['h']


class TFEmbedding(nn.Block):
    def __init__(self, embedding_size, dropout):
        super(TFEmbedding, self).__init__()

        seq = nn.Sequential()
        with seq.name_scope():
            seq.add(nn.Dense(embedding_size, use_bias=True))  # 全连接子层
            seq.add(nn.Dropout(dropout))
        self.proj_TF = seq

    def forward(self, ndata):
        extra_repr = self.proj_TF(ndata['TF_features'])

        # forward()方法中，使用前面定义的全连接层和容器seq对输入的节点数据ndata中的
        # TF_features进行处理，从而生成对应的向量表示。然后将这些向量作为结果返回。
        return extra_repr


class tgEmbedding(nn.Block):
    def __init__(self, embedding_size, dropout):
        super(tgEmbedding, self).__init__()

        seq = nn.Sequential()
        with seq.name_scope():
            seq.add(nn.Dense(embedding_size, use_bias=True))
            seq.add(nn.Dropout(dropout))
        self.proj_tg = seq

    def forward(self, ndata):
        extra_repr = self.proj_tg(ndata['tg_features'])
        return extra_repr



# 计算TF和tg之间相关性得分
# featuer_size 表示输入的两个向量的特征维度的大小
class BilinearDecoder(nn.Block):
    def __init__(self, feature_size):
        super(BilinearDecoder, self).__init__()


        self.activation = nn.Activation('sigmoid')
        with self.name_scope():     # 定义权重矩阵
            self.W = self.params.get('dot_weights', shape=(feature_size, feature_size))

    def forward(self, h_TF, h_tg):
        
        results_mask = self.activation((nd.dot(h_TF, self.W.data()) * h_tg).sum(1))

        return results_mask

