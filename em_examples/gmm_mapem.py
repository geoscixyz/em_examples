import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
from sklearn.mixture.gaussian_mixture import (
    _compute_precision_cholesky
)
import copy
import seaborn
import ipywidgets

from .Base import widgetify
from .gmmutils import *

n = 2

def GMM1D_reference(mean0, mean1, std0, std1, weight0):

    means = np.r_[mean0, mean1]
    sigma = np.r_[std0**2., std1**2.]
    clfref = GaussianMixture(
        n_components=n,
        covariance_type='full',
        reg_covar=1e-6,
        max_iter=100,
        tol=1e-3
    )
    clfref.fit(means[:, np.newaxis])
    clfref.weights_ = np.r_[weight0, 1. - weight0]
    clfref.means_ = means[:, np.newaxis]
    clfref.covariances_ = sigma.reshape((2, 1, 1))
    clfref.precisions_cholesky_ = _compute_precision_cholesky(
        clfref.covariances_, clfref.covariance_type
    )
    computePrecision(clfref)
    order_clusters_GM_weight(clfref)

    return clfref


def GMM1D_plot_reference_app(mean0, mean1, std0, std1, weight0):

    clf = GMM1D_reference(mean0, mean1, std0, std1, weight0)

    xmin, xmax = np.min(clf.means_ - 5. * np.sqrt(clf.covariances_)
                        ), np.max(clf.means_ + 5. * np.sqrt(clf.covariances_))
    dx = (xmax - xmin) / 1000

    x = np.linspace(xmin, xmax, 1000)

    rv0 = multivariate_normal(mean0, std0**2.)
    rv1 = multivariate_normal(mean1, std1**2.)

    fig = plt.figure(figsize=(8, 3))
    ax = fig.add_subplot(111)

    ax.plot(x, np.exp(clf.score_samples(
        x[:, np.newaxis])), linewidth=3., linestyle='-', color='gray', label='$\Theta_{prior}$')
    ax.plot(x, weight0 * rv0.pdf(x), linestyle='--',
            color='blue', alpha=0.5, label='Component 1')
    ax.plot(x, (1. - weight0) * rv1.pdf(x), linestyle='dashdot',
            color='red', alpha=0.5, label='Component 2')
    ax.legend(fontsize=16, loc=0)
    ax.set_xlabel('Physical Property', fontsize=20)
    ax.set_ylabel('Probability Density', fontsize=20)
    ax.tick_params(labelsize=14)
    # ax.set_xlim([-8,8])
    plt.tight_layout()
    plt.show()


def GMM1D_define_reference_app():
    app = widgetify(
        GMM1D_plot_reference_app,
        manual=False,
        mean0=ipywidgets.FloatSlider(
            value=-5,
            min=-10.,
            max=10.0,
            step=0.1,
            description='Mean Cluster 1',
            style={'description_width': 'initial'},
            layout=ipywidgets.Layout(width='50%')
        ),
        mean1=ipywidgets.FloatSlider(
            value=5,
            min=-10.,
            max=10.0,
            step=0.1,
            description='Mean Cluster 2',
            style={'description_width': 'initial'},
            layout=ipywidgets.Layout(width='50%')
        ),
        std0=ipywidgets.FloatSlider(
            value=1,
            min=0.1,
            max=10.0,
            step=0.1,
            description='Std-Dev Cluster 1',
            style={'description_width': 'initial'},
            layout=ipywidgets.Layout(width='50%')
        ),
        std1=ipywidgets.FloatSlider(
            value=2,
            min=0.1,
            max=10.0,
            step=0.1,
            description='Std-Dev Cluster 2',
            style={'description_width': 'initial'},
            layout=ipywidgets.Layout(width='50%')
        ),
        weight0=ipywidgets.FloatSlider(
            value=0.5,
            min=0,
            max=1,
            step=0.1,
            description='Weight cluster 1',
            style={'description_width': 'initial'},
            layout=ipywidgets.Layout(width='50%')  # , height='80px')
        ),
    )

    app.clf = GMM1D_reference(
        app.children[0].value,
        app.children[1].value,
        app.children[2].value,
        app.children[3].value,
        app.children[4].value)

    return app


def MAPEM_fitting(clf, clfref, alpha, kappa, nu):

    xupdate, _ = clf.sample(10000)

    clfupdate = GaussianMixtureWithPrior(
        GMref=clfref,
        alphadir=alpha * np.ones(clfref.n_components),
        kappa=kappa * np.ones(clfref.n_components),
        nu=nu * np.ones(clfref.n_components),
        verbose=False,
        prior_type='semi',
        max_iter=100,
        n_init=1,
        reg_covar=1e-6,
        weights_init=clf.weights_,
        means_init=clf.means_,
        precisions_init=clf.precisions_,
        tol=1e-3,
        update_covariances=True,
    )

    clfupdate = clfupdate.fit(xupdate)
    order_clusters_GM_weight(clfupdate)

    return xupdate, clfupdate


def GMM1D_MAPEM(clfref, mean0, mean1, std0, std1, weight0, alpha, kappa, nu):

    clf = GMM1D_reference(mean0, mean1, std0, std1, weight0)
    xupdate, clfupdate = MAPEM_fitting(clf, clfref, alpha, kappa, nu)

    xmin0, xmax0 = np.min(clfref.means_ - 5. * np.sqrt(clfref.covariances_)
                          ), np.max(clfref.means_ + 5. * np.sqrt(clfref.covariances_))
    xmin1, xmax1 = np.min(clf.means_ - 5. * np.sqrt(clf.covariances_)
                          ), np.max(clf.means_ + 5. * np.sqrt(clf.covariances_))
    xmin = np.min([xmin0, xmin1])
    xmax = np.max([xmax0, xmax1])
    dx = (xmax - xmin) / 1000

    x = np.linspace(xmin, xmax, 1000)

    rv0 = multivariate_normal(mean0, std0**2.)
    rv1 = multivariate_normal(mean1, std1**2.)

    fig = plt.figure(figsize=(8, 3))
    ax = fig.add_subplot(111)

    ax.plot(x, np.exp(clfref.score_samples(
        x[:, np.newaxis])), linewidth=1., linestyle='--', color='gray', label='$\Theta_{prior}$')
    ax.plot(x, np.exp(clfupdate.score_samples(
        x[:, np.newaxis])), linewidth=2., linestyle='-', color='k', label='$\Theta_{MAP-EM}$')
    ax.plot(x, np.exp(clf.score_samples(
        x[:, np.newaxis])), linewidth=1., linestyle='-', color='blue', label='$\Theta_{samples}$')
    ax.hist(xupdate, bins=100, density=True, color='blue', alpha=0.3, label='Samples set')
    ax.legend(fontsize=16, loc=0)
    ax.set_xlabel('Physical Property', fontsize=20)
    ax.set_ylabel('Probability Density', fontsize=20)
    ax.tick_params(labelsize=14)
    # ax.set_xlim([-8,8])
    plt.tight_layout()
    plt.show()


def GMM1D_MAPEM_app(clfref):

    app = widgetify(
        GMM1D_MAPEM,
        manual=False,
        clfref=ipywidgets.fixed(clfref),
        mean0=ipywidgets.FloatSlider(
            value=-5,
            min=-10.,
            max=10.0,
            step=0.1,
            description='Mean Cluster 1',
            style={'description_width': 'initial'},
            layout=ipywidgets.Layout(width='50%')
        ),
        mean1=ipywidgets.FloatSlider(
            value=5,
            min=-10.,
            max=10.0,
            step=0.1,
            description='Mean Cluster 2',
            style={'description_width': 'initial'},
            layout=ipywidgets.Layout(width='50%')
        ),
        std0=ipywidgets.FloatSlider(
            value=1,
            min=0.1,
            max=10.0,
            step=0.1,
            description='Std-Dev Cluster 1',
            style={'description_width': 'initial'},
            layout=ipywidgets.Layout(width='50%')
        ),
        std1=ipywidgets.FloatSlider(
            value=2,
            min=0.1,
            max=10.0,
            step=0.1,
            description='Std-Dev Cluster 2',
            style={'description_width': 'initial'},
            layout=ipywidgets.Layout(width='50%')
        ),
        weight0=ipywidgets.FloatSlider(
            value=0.5,
            min=0,
            max=1,
            step=0.1,
            description='Weight cluster 1',
            style={'description_width': 'initial'},
            layout=ipywidgets.Layout(width='50%')  # , height='80px')
        ),
        alpha=ipywidgets.FloatSlider(
            value=1.,
            min=0.,
            max=100.0,
            step=0.1,
            description='Alpha (weights confidence)',
            style={'description_width': 'initial'},
            layout=ipywidgets.Layout(width='50%')
        ),
        kappa=ipywidgets.FloatSlider(
            value=1.,
            min=0.,
            max=100.0,
            step=0.1,
            description='Kappa (means confidence)',
            style={'description_width': 'initial'},
            layout=ipywidgets.Layout(width='50%')
        ),
        nu=ipywidgets.FloatSlider(
            value=1.,
            min=0.,
            max=100.0,
            step=0.1,
            description='Nu (covariances confidence)',
            style={'description_width': 'initial'},
            layout=ipywidgets.Layout(width='50%')
        ),
    )

    return app
