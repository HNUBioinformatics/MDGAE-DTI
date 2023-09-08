import numpy as np
import pandas as pd
import mxnet as mx
from mxnet import ndarray as nd
import dgl


# load_data加载数据
def load_data(directory):
    # chemical_similarity as fesddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddature
    TFSM_1 = np.loadtxt('', delimiter=",")  
    tgSM_1 = np.loadtxt('', delimiter=",") 

    tg_TF_SM_1 = np.loadtxt('')
    TF_tg_SM_1 = tg_TF_SM_1.T                 

    # seq_similarity as feature
    TFSM_2 = np.loadtxt('')   
    tgSM_2 = np.loadtxt('')  
    TF_tg_SM_2 = TF_tg_SM_1      
    tg_TF_SM_2 = TF_tg_SM_2.T    
    
    return TFSM_1, tgSM_1, TF_tg_SM_1, tg_TF_SM_1, TFSM_2, tgSM_2, TF_tg_SM_2, tg_TF_SM_2


# 抽样
def sample(directory, random_seed):
    all_associations = pd.read_csv('', names=['TF', 'tg', 'label'])
    known_associations = all_associations.loc[all_associations['label'] == 1]
    unknown_associations = all_associations.loc[all_associations['label'] == 0]

    random_negative = unknown_associations.sample(n=known_associations.shape[0], random_state=random_seed, axis=0)  

    sample_df = known_associations.append(random_negative)
    sample_df.reset_index(drop=True, inplace=True)    inplace为True表示直接在原对象上修改

    #all_associations.reset_index(drop=True, inplace=True)

    return sample_df.values  

# 构建包含节点特征和边特征的图
def build_graph(directory, random_seed, ctx):
    # dgl.load_backend('mxnet')
    # load_data从指定文件夹中读取药物靶标相互作用的关系数据集，用于构建图神经网络
    TFSM, tgSM, TF_tg_SM, tg_TF_SM, TFSM_2, tgSM_2, TF_tg_SM_2, tg_TF_SM_2 = load_data(directory)
    samples = sample(directory, random_seed)

    print('Building graph ...')
    g1 = dgl.DGLGraph(multigraph=True)  # g1包含所有节点  multigraph=True可以存在多重边
    # TFSM.shape[1] 获取矩阵元组中的第二个元素，矩阵的列数
    g1.add_nodes(TFSM.shape[1] + tgSM.shape[1])  # 添加节点 并且节点总数是 药物节点数量+靶标节点数量
    # 创建一个初值为0的一维数组，数组长度是 g1中的节点数量
    node_type = nd.zeros(g1.number_of_nodes(), dtype='float32', ctx=ctx)  # 用来表示节点类型的一维数组
    node_type[:TFSM.shape[1]] = 1   # 前TFSM.shape[1]个节点类型设置为1，表示药物节点
    g = g1.to(ctx)   # 将g1转换成ctx下的 对象g
    g.ndata['type'] = node_type  # 将节点类型数据添加到图中

    # concate features
    print('Adding TF features ...')
    # TF_data = nd.zeros(shape=(g.number_of_nodes(), TFSM.shape[1]+TFSM_2.shape[1]), dtype='float32', ctx=ctx)
    # TF_data[:TFSM.shape[0], :TFSM.shape[1]] = nd.from_numpy(TFSM)
    # TF_data[:TFSM.shape[0],TFSM.shape[1]:TFSM.shape[1]+TFSM_2.shape[1]] = nd.from_numpy(TFSM_2)
    # TF_data[TFSM.shape[0]: TFSM.shape[0]+tgSM.shape[0], :tg_TF_SM.shape[1]] = nd.from_numpy(tg_TF_SM)
    # TF_data[TFSM.shape[0]: TFSM.shape[0]+tgSM.shape[0], tg_TF_SM.shape[1]:tg_TF_SM.shape[1]+tg_TF_SM_2.shape[1]] = nd.from_numpy(tg_TF_SM_2)
    dr_mat = np.hstack((TFSM,TFSM_2))  # 将TFSM和TFSM_2按照列进行拼接  拼接药物化学结构相似性和药物相似性
    dr_mat1 = np.hstack((tg_TF_SM, tg_TF_SM_2))   # 拼接药物靶标相似性
    TF_data = np.vstack((dr_mat, dr_mat1))        # 按行拼接
    TF_data = nd.array(TF_data, dtype='float32', ctx=ctx)  # 转换TF_data为mxnet NDArray数据类型

    print(TF_data.shape)
    # 为图中每个（药物）节点添加一个“TF_features”特征，并将该特征赋值为TF_data
    # 将TF_data赋值给 图G中所有节点的TF_features特征
    # 将节点特征和图结构进行关联
    g.ndata['TF_features'] = TF_data

    print('Adding target gene features ...')
    # tg_data = nd.zeros(shape=(g.number_of_nodes(), tgSM.shape[1]+tgSM_2.shape[1]), dtype='float32', ctx=ctx)
    # tg_data[:TFSM.shape[0], :TF_tg_SM.shape[1]] = nd.from_numpy(TF_tg_SM)
    # tg_data[:TFSM.shape[0],TF_tg_SM.shape[1]:TF_tg_SM.shape[1]+TF_tg_SM_2.shape[1]] = nd.from_numpy(TF_tg_SM_2)
    # tg_data[TFSM.shape[0]: TFSM.shape[0]+tgSM.shape[0], :tgSM.shape[1]] = nd.from_numpy(tgSM)
    # tg_data[TFSM.shape[0]: TFSM.shape[0]+tgSM.shape[0], tgSM.shape[1]:tgSM.shape[1]+tgSM_2.shape[1]] = nd.from_numpy(tgSM_2)

    tg_mat = np.hstack((TF_tg_SM, TF_tg_SM_2))

    tg_mat1 = np.hstack((tgSM, tgSM_2))

    tg_data = np.vstack((tg_mat, tg_mat1))
    tg_data = nd.array(tg_data, dtype='float32', ctx=ctx)



    print(tg_data.shape)
    g.ndata['tg_features'] = tg_data

    print('Adding edges ...')
    # TFSM.shape[1] 表示矩阵第二维度的大小，即列数
    TF_ids = list(range(1, TFSM.shape[1] + 1))  # 生成一个列表 从1到TFSM.shape[1] + 1
    tg_ids = list(range(1, tgSM.shape[1]+1))
    # 将（图中的节点名称：即TF_ids中的元素）TF_ids映射为对应节点在DGL图数据结构中的整数ID（即列表中元素对应的索引值）
    TF_ids_invmap = {id_: i for i, id_ in enumerate(TF_ids)}  # 生成一个字典TF_ids_invmap
    tg_ids_invmap = {id_: i for i, id_ in enumerate(tg_ids)}

    sample_TF_vertices = [TF_ids_invmap[id_] for id_ in samples[:, 0]]  # 生成列表sample_TF_vertices；样本中所有药物节点在DGL图中的整数ID构成的一维数组
    sample_tg_vertices = [tg_ids_invmap[id_] + TFSM.shape[0] for id_ in samples[:, 1]]

    g.add_edges(sample_TF_vertices, sample_tg_vertices,
                data={'inv': nd.zeros(samples.shape[0], dtype='int32', ctx=ctx),
                      'rating': nd.from_numpy(samples[:, 2].astype('float32')).copyto(ctx)})
    
    g.add_edges(sample_tg_vertices, sample_TF_vertices,
                data={'inv': nd.zeros(samples.shape[0], dtype='int32', ctx=ctx),
                      'rating': nd.from_numpy(samples[:, 2].astype('float32')).copyto(ctx)})
                      
    g.readonly()
    print('Successfully build graph !!')

    return g, TF_ids_invmap, tg_ids_invmap, TFSM

