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

def plot_losses(d_real_losses=None, d_fake_losses=None, 
                g_losses=None, e_losses=None, savename=None, 
                title=None, smooth=False, itv=None, marker_size=5):
    assert d_real_losses is not None
    assert d_fake_losses is not None
    assert g_losses is not None
    assert e_losses is not None
    
    plt.figure()
    plt.xlabel("Iterations")
    plt.ylabel("Loss")

    #if title is not None:
    #    plt.title(title)
    plt.title("Loss Plot")
    X = np.arange(len(e_losses))
    if smooth:
        Xnew = np.linspace(0, len(e_losses) - 1, 300)
        d_real_losses = spline(X, d_real_losses, Xnew)
        d_fake_losses = spline(X, d_fake_losses, Xnew)
        g_losses = spline(X, g_losses, Xnew)
        e_losses = spline(X, e_losses, Xnew)
        X = Xnew
    if itv is None:
        itv = len(X) // 200
    plt.plot(X[::itv], d_real_losses[::itv], label="D loss reals")
    plt.plot(X[::itv], d_fake_losses[::itv], label="D loss fakes")
    plt.plot(X[::itv], g_losses[::itv], label="G loss")
    plt.plot(X[::itv], e_losses[::itv], label="E loss")
    
    plt.legend()
    plt.tight_layout()
    plt.show()
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
        plot_losses(**X)
    if len(sys.argv) > 2:
        f = sys.argv[2]
        X = np.load(f)
        plot_latent_encodings(X)

