import os

import faiss
import matplotlib.pyplot as plt
import numpy as np

def mine_nearest_neighbors(features, real_label, topk=20, calculate_accuracy=False):
    # mine the topk nearest neighbors for every sample
    features = features
    n, dim = features.shape[0], features.shape[1]
    # print("n = ", n, ", dim = ", dim)
    index = faiss.IndexFlatIP(dim)
    # index = faiss.index_cpu_to_all_gpus(index)
    index.add(features)
    distances, indices = index.search(features, topk + 1)  # Sample itself is included

    # print(distances.shape, indices.shape)
    # print("raw: ", distances[:2], indices[:5])
    # evaluate
    if calculate_accuracy:
        targets = real_label
        neighbor_targets = np.take(targets, indices[:, 1:], axis=0)  # Exclude sample itself for eval
        anchor_targets = np.repeat(targets.reshape(-1, 1), topk, axis=1)
        # print(neighbor_targets.shape, anchor_targets.shape)
        # print("label: ", neighbor_targets[:2], anchor_targets[:2])
        accuracy = np.mean(neighbor_targets == anchor_targets)
        return indices, accuracy
    else:
        return indices

def plt_knn_acc(high_acc, low_acc, dataset_name, pic_n=""):
    x = np.arange(1, len(high_acc)+1, 1)
    # plt.plot(x, high_acc, label="high dim")
    plt.plot(x, low_acc, label="low dim")
    plt.xlabel("epoch")
    plt.ylabel("k near accuracy")
    plt.ylim(0, 1)
    plt.title(dataset_name+pic_n+"_cls")
    plt.legend()
    # plt.show()
    current_path = os.path.abspath(__file__)
    father_path = os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".")
    print("father_path = ", father_path)
    path = father_path + '/pic_result/' + dataset_name + pic_n+'_cls.png'
    plt.savefig(path)
    plt.close()