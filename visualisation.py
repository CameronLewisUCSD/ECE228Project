# in this file we will place our funciton related to graphing losses and other information from the train file
from matplotlib.pylab import subplots
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from matplotlib.patches import ConnectionPatch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import torch
import pandas as pd
from cuml import TSNE

def get_graphs(val_loss, train_loss, path):
    fig,axs=subplots(1,1)

    #Plot accuracy vs epoch
    # plt.subplot(121)
    # plt.figsize(15,15)

    #### Fill in plot ####


    #Plot loss vs epoch
    ax=axs
    ax.plot(train_loss,color='g')
    ax.plot(val_loss,color='r')
    ax.legend(['Train_loss', 'Val_loss'])
    ax.set_title('Loss vs Epoch (AutoEncoder)')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epoch')


    fig.set_figheight(10)
    fig.set_figwidth(10)
    fig.savefig(path)
    
def tsne(data):
    '''Perform t-SNE on data.
    Parameters
    ----------
    data : array (M,N)
    Returns
    -------
    results : array (M,2)
        2-D t-SNE embedding
    '''
    print('Running t-SNE...', end="", flush=True)
    M = len(data)
    np.seterr(under='warn')
    results = TSNE(
        n_components=2,
        perplexity=int(M/100),
        early_exaggeration=20,
        learning_rate=int(M/12),
        n_iter=2000,
        verbose=0,
        random_state=2009
    ).fit_transform(data.astype('float64'))
    print('complete.')
    return results
#Create graph
def view_TSNE(results, labels, title, show=False):
    label_list, counts = np.unique(labels, return_counts=True)

    textsize = 14
    colors = cmap_lifeaquatic(len(counts))
    data = np.stack([(labels+1), results[:,0], results[:,1]], axis=1)
    df = pd.DataFrame(data=data, columns=["Class", "x", "y"])
    df["Class"] = df["Class"].astype('int').astype('category')

    fig = plt.figure(figsize=(6,8))
    params = {'mathtext.default': 'regular' }
    plt.rcParams.update(params)
    gs = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[3, 1])

    ax1 = fig.add_subplot(gs[0])
    for j in range(len(df["Class"].cat.categories)):
        plt.plot(df[df.Class == df["Class"].cat.categories[j]].x, df[df.Class == df["Class"].cat.categories[j]].y, 'o', alpha=0.2, c=colors[j], ms=6, mec="w", mew=0.5, rasterized=True, label=j+1)
    plt.axis('off')
    leg = plt.legend(loc='center left', bbox_to_anchor=(0.9, 0.75), ncol=1, title="Class", title_fontsize=textsize)
    for lh in leg.legendHandles:
        lh._legmarker.set_alpha(1)
    plt.title(title, fontsize=textsize)

    ax2 = fig.add_subplot(gs[1])
    arr = plt.hist(labels+1, bins=np.arange(1, max(labels)+3, 1), histtype='bar', align='left', rwidth=0.8, color='k')
    plt.grid(axis='y', linestyle='--')
    plt.xticks(label_list+1, label_list+1)
#     plt.ylim([0, 1.25 * max(counts)])
    plt.ylim([0, 30000])
    ax2.set_xlabel('Class', fontsize=textsize)
    ax2.set_ylabel('Detections', fontsize=textsize)
    plt.title('Class Assignments, $N_{train}$ = ' + f'{len(labels)}', fontsize=textsize)

    N = counts.sum()
    def CtP(x):
        return 100 * x / N

    def PtC(x):
        return x * N / 100

    ax3 = ax2.secondary_yaxis('right', functions=(CtP, PtC))
    ax3.set_ylabel('$\%N_{train}$', fontsize=textsize)
    for i in range(len(np.unique(labels))):
        plt.text(arr[1][i], 1.05 * arr[0][i], str(int(arr[0][i])), ha='center')

    fig.subplots_adjust(left=0.15, right=0.9)

    if show:
        plt.show()
    else:
        plt.close()
    return fig

def cmap_lifeaquatic(N=None):
    """
    Returns colormap inspired by Wes Andersen's The Life Aquatic
    Available from https://jiffyclub.github.io/palettable/wesanderson/
    """
    colors = [
        (27, 52, 108),
        (244, 75, 26),
        (67, 48, 34),
        (35, 81, 53),
        (123, 109, 168),
        (139, 156, 184),
        (214, 161, 66),
        (1, 170, 233),
        (195, 206, 208),
        (229, 195, 158),
        (56, 2, 130),
        (0, 0, 0)
    ]
    colors = [tuple([v / 256 for v in c]) for c in colors]
    if colors is not None:
        return colors[0:N]
    else:
        return colors