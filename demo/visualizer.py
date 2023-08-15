import torch
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import numpy as np

def plot_TSNE(features_all, proposals_all, mem, class_labels):
    tsne = TSNE(init='pca', learning_rate='auto')  # init='random' and lr=200.0 is default

    features_all = torch.cat(features_all, dim=0).cpu()
    proposals_all = torch.cat([b.bbox for b in proposals_all], dim=0)
    mem_data = mem.cpu()

    # find index by calculating minimum distance
    dists, indexes = torch.cdist(mem_data, features_all).min(dim=-1)
    # 모델의 출력값을 tsne.fit_transform에 입력하기
    pred_tsne = tsne.fit_transform(features_all)
    mem_data_tsne = pred_tsne[indexes]

    plt.cla()
    # plot all data points sampled
    xs = pred_tsne[:, 0]
    ys = pred_tsne[:, 1]
    plt.scatter(xs, ys, c='gray', marker='.')
    # plot all data points memorized
    xs = mem_data_tsne[:, 0]
    ys = mem_data_tsne[:, 1]
    label_names = ['bg',  # always index 0
                   'airplane', 'antelope', 'bear', 'bicycle', 'bird',
                   'bus', 'car', 'cattle', 'dog', 'domestic_cat',
                   'elephant', 'fox', 'giant_panda', 'hamster', 'horse',
                   'lion', 'lizard', 'monkey', 'motorcycle', 'rabbit',
                   'red_panda', 'sheep', 'snake', 'squirrel', 'tiger',
                   'train', 'turtle', 'watercraft', 'whale', 'zebra']
    colors = ['black',  # always index 0
              'brown', 'chocolate', 'gold', 'khaki', 'olive',
              'greenyellow', 'darkolivegreen', 'darkseagreen', 'springgreen', 'r',
              'darkseagreen', 'cyan', 'deepskyblue', 'skyblue', 'steelblue',
              'powderblue', 'royalblue', 'navy', 'blue', 'slateblue',
              'darkslateblue', 'blueviolet', 'thistle', 'magenta', 'deeppink',
              'hotpink', 'crimson', 'pink', 'lightcoral', 'rosybrown']
    # colors = cm.prism(np.linspace(0, 1, 31))
    for cls, c in enumerate(colors):
        tf_cls = class_labels == cls
        # plt.scatter(xs[tf_cls], ys[tf_cls], label=label_names[cls], c=c, cmap=plt.cm.rainbow, marker='.')
        plt.scatter(xs[tf_cls], ys[tf_cls], label=label_names[cls], c=colors[cls], marker='.')
    plt.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), fontsize='xx-small')
    # plt.scatter(xs, ys, label=label_names[class_labels], c=class_labels, cmap=plt.cm.get_cmap('rainbow', 31), marker='.')
    # plt.colorbar(ticks=label_names)
    plt.savefig('tsne.png')
    plt.show()

def plot_histogram(contrib_list, l2_norms_list):
    plt.cla()
    contrib = torch.cat(contrib_list).cpu()
    l2_norms = torch.cat(l2_norms_list).cpu()
    targets_df = pd.DataFrame({'Contrib': contrib,
                               'L2_norm': l2_norms})
    sns.kdeplot(targets_df[
                    'Contrib'])  # https://mindscale.kr/course/python-visualization-basic/dist/, https://coding-kindergarten.tistory.com/132
    # sns.ecdfplot(targets_df['contrib'])
    plt.show()
    plt.cla()
    sns.kdeplot(targets_df['L2_norm'])
    # sns.ecdfplot(targets_df['contrib'])
    plt.show()


def contrib_L2_plot(contrib, l2_norms, name, name_contrib):
    plt.cla()
    keep_irrelevant = (torch.softmax(l2_norms, dim=0) > 1 / len(l2_norms))
    plt.scatter(contrib, l2_norms, c=keep_irrelevant, cmap=plt.cm.rainbow, marker='.')
    plt.title(name, fontsize=20)
    #major_ticks_topx = np.linspace(0, 1, 11)
    #minor_ticks_topy = np.linspace(0.6, 2.4, 10)
    #plt.xticks(major_ticks_topx)
    #plt.yticks(minor_ticks_topy)
    plt.xlabel("contrib_" + name_contrib, fontsize=20)
    plt.ylabel("l2_norm", fontsize=20)
    plt.show()


def contrib_L2_plots(contrib_list, l2_norms_list, name):
    contrib = torch.cat(contrib_list).cpu()
    l2_norms = torch.cat(l2_norms_list).cpu()
    contrib_L2_plot(contrib, l2_norms, name, 'all')
    contrib_mean = torch.stack(contrib_list).mean(dim=0).cpu()
    l2_norms_mean = torch.stack(l2_norms_list).mean(dim=0).cpu()
    contrib_L2_plot(contrib_mean, l2_norms_mean, name, 'mean')
    contrib_max = torch.stack(contrib_list).max(dim=0)[0].cpu()
    l2_norms_max = torch.stack(l2_norms_list).max(dim=0)[0].cpu()
    contrib_L2_plot(contrib_max, l2_norms_max, name, 'max')
