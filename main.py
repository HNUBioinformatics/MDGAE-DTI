import matplotlib.pyplot as plt
import mxnet as mx
import numpy as np
import pandas as pd
from mxnet import ndarray as nd
from scipy import interp
from sklearn import metrics
import warnings, sys
import networkx as nx
from train import Train



if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    auc, acc, pre, recall, f1, fprs, tprs,embedding = Train(directory='data',
                                                  epochs=150,
                                                  aggregator='GATConv',

                                                  embedding_size=64,
                                                  layers=4,
                                                  dropout=0.2,
                                                  slope=0.2,  # LeakyReLU
                                                  lr=0.0005,
                                                  wd=1e-3,
                                                  random_seed=1,
                                                  ctx=mx.cpu(0))

    # 计算模型的AUPR
    PR_dot_matrix = np.mat(sorted(np.column_stack((recall, pre)).tolist())).T
    PR_dot_matrix.T[0] = [0, 1]
    PR_dot_matrix = np.c_[PR_dot_matrix, [1, 0]]
    x_PR = PR_dot_matrix[0].T
    y_PR = PR_dot_matrix[1].T
    aupr = 0.5 * (x_PR[1:] - x_PR[:-1]).T * (y_PR[:-1] + y_PR[1:])
  
    # aupr1 = auc(np.array(recall),np.array(pre))
    print('AUC mean: %.4f, variance: %.4f \n' % (np.mean(auc), np.std(auc)),
          'Accuracy mean: %.4f, variance: %.4f \n' % (np.mean(acc), np.std(acc)),
          'Precision mean: %.4f, variance: %.4f \n' % (np.mean(pre), np.std(pre)),
          'Recall mean: %.4f, variance: %.4f \n' % (np.mean(recall), np.std(recall)),
          'F1-score mean: %.4f, variance: %.4f \n' % (np.mean(f1), np.std(f1)),
          'AUPR mean: %.4f \n' % (aupr))
   # print(aupr1)
    mean_fpr = np.linspace(0, 1, 10000)


    tpr = []
    plt.switch_backend("Agg")
    for i in range(len(fprs)):
        tpr.append(interp(mean_fpr, fprs[i], tprs[i]))
        tpr[-1][0] = 0.0
        plt.plot(fprs[i], tprs[i], alpha=0.4, label='ROC fold %d (AUC = %.4f)' % (i + 1, auc[i]))

    mean_tpr = np.mean(tpr, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = metrics.auc(mean_fpr, mean_tpr)
    auc_std = np.std(auc)
    plt.plot(mean_fpr, mean_tpr, color='b', alpha=0.8, label='Mean AUC (AUC = %.4f $\pm$ %.4f)' % (mean_auc, auc_std))

    std_tpr = np.std(tpr, axis=0)
    tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
    tpr_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tpr_lower, tpr_upper, color='grey', alpha=0.3, label='$\pm$ 1 std.dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc='lower right')
    plt.show()
    plt.tight_layout() 
    plt.savefig('./4.png',dpi = 1000,edgecolor = 'lightgreen', bbox_inches = 'tight')
   
   # plt.xlabel('Recall')
   # plt.ylabel('Precision')
   # plt.grid()  # 生成网格
   # plt.plot(recall, pre)
   # plt.figure("P-R Curve")
   # plt.savefig('./32.png', dpi=1000, edgecolor='lightgreen', bbox_inches='tight')

    print("程序结束")
    sys.exit()
    
