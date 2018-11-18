try:
    import matplotlib.pyplot as plt
    plt.switch_backend("agg")
except ImportError:
    import matplotlib as mpl
    mpl.use('TkAgg')
    import matplotlib.pyplot as plt

import numpy as np 
from scipy.interpolate import spline
import sys
import re

def plot_losses(d_real_losses=None, d_fake_losses=None, 
                g_losses=None, e_losses=None, savename=None, 
                title=None, smooth=True):

    assert d_real_losses is not None
    assert d_fake_losses is not None
    assert g_losses is not None
    assert e_losses is not None
    
    plt.figure()
    plt.xlabel("Iterations")
    plt.ylabel("Loss")

    if title is not None:
        plt.title(title)

    X = np.arange(len(e_losses))
    if smooth:
        Xnew = np.linspace(0, len(e_losses) - 1, 300)
        d_real_losses = spline(X, d_real_losses, Xnew)
        d_fake_losses = spline(X, d_fake_losses, Xnew)
        g_losses = spline(X, g_losses, Xnew)
        e_losses = spline(X, e_losses, Xnew)
        X = Xnew

    plt.plot(X, d_real_losses, label="D loss reals")
    plt.plot(X, d_fake_losses, label="D loss fakes")
    plt.plot(X, g_losses, label="G loss")
    plt.plot(X, e_losses, label="E loss")
    
    plt.legend()
    plt.tight_layout()
    try:
        plt.show()
    except:
        pass
    if savename is not None:
        plt.savefig(savename)
        print("Plot saved to %s" % savename)

if __name__ == "__main__":
    f = sys.argv[1]
    if len(sys.argv) == 3:
        savename = sys.argv[2]
    else:
        savename = None
    X = np.load(f)
    plot_losses(savename=savename, **X)
