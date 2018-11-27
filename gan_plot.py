try:
    import matplotlib.pyplot as plt
    plt.switch_backend("agg")
except ImportError:
    import matplotlib as mpl
    mpl.use('TkAgg')
    import matplotlib.pyplot as plt

import sys
import numpy as np 
from scipy.interpolate import spline
from sklearn.manifold import TSNE
import re

def plot_losses(loss_dict, savename=None, 
                title=None, itv=None, marker_size=5):
    
    plt.figure()
    plt.xlabel("Iterations")
    plt.ylabel("Loss")

    #if title is not None:
    #    plt.title(title)
    plt.title("Loss Plot")
    X = np.arange(len(list(loss_dict.values())[0]))
    if itv is None:
        itv = len(X) // 200
    for k, v in loss_dict.items():
        plt.plot(X[::itv], v[::itv], label=k)

    
    plt.legend()
    plt.tight_layout()
    try:
        plt.show()
    except:
        pass
    if savename is not None:
        plt.savefig(savename)
        print("Losses plot saved to %s" % savename)

def plot_latent_encodings(latent_encodings, savename=None):
    plt.figure()
    plt.title("Latent Encodings")
    for i,label_inputs in enumerate(latent_encodings):
        label_embeddings = TSNE().fit_transform(label_inputs)
        plt.scatter(label_embeddings[:,0], label_embeddings[:,1], label=i)
    plt.xlabel("L1")
    plt.ylabel("L2")
    plt.legend()
    try:
        plt.show()
    except:
        pass
    if savename is not None:
        plt.savefig(savename)
        print("Latent encoding plots saved to %s" % savename)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        f = sys.argv[1]
        X = np.load(f)
        plot_losses(X)
    if len(sys.argv) > 2:
        f = sys.argv[2]
        X = np.load(f)
        plot_latent_encodings(X)

