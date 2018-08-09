import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import copy
import seaborn
import ipywidgets

from .Base import widgetify

seaborn.set()
matplotlib.rcParams['font.size'] = 14


def rotationmatrix(angle):
    rad = np.radians(angle)
    R = np.r_[[[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]]]
    return R


def GMM2D_creation(n, mean0, mean1, scale0, scale1, rotation, nbsamples=1000,
                   weights=None, dxy=1e3, fontsize=16, ncontour=10):

    assert len(mean0) == n, 'mean0 should be length {}'.format(n)
    assert len(mean1) == n, 'mean1 should be length {}'.format(n)
    assert len(scale0) == n, 'scale0 should be length {}'.format(n)
    assert len(scale1) == n, 'scale1 should be length {}'.format(n)
    assert len(rotation) == n, 'rotation should be length {}'.format(n)

    if weights is None:
        weights = np.ones(n) / n
    elif np.sum(weights) == 0.:
        weights = np.ones(n) / n
    else:
        weights /= np.sum(weights)

    xmin, xmax = np.min(mean0 - 5. * scale0), np.max(mean0 + 5. * scale0)
    dx = (xmax - xmin) / dxy
    ymin, ymax = np.min(mean1 - 5. * scale1), np.max(mean1 + 5. * scale1)
    dy = (ymax - ymin) / dxy

    testXplot0 = np.linspace(xmin, xmax, 1000)
    testXplot1 = np.linspace(ymin, ymax, 1000)

    x, y = np.mgrid[xmin:xmax:dx, ymin:ymax:dy]
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x
    pos[:, :, 1] = y
    mean = np.r_[0., 0.]
    sigma = np.r_[[1., 05.], [0.5, 1]]
    gaussianlist = []
    gaussian1dlist0 = []
    gaussian1dlist1 = []
    samplelist = []
    sample1dlist0 = []
    sample1dlist1 = []
    for i in range(n):
        mean = np.r_[mean0[i], mean1[i]]
        R = rotationmatrix(rotation[i])
        S = np.diag([scale0[i], scale1[i]])
        Sigma = R.T.dot(S.T.dot(S.dot(R)))
        rv = multivariate_normal(mean, Sigma)
        sample = rv.rvs(int(weights[i] * nbsamples))
        gaussianlist.append(copy.deepcopy(rv))
        samplelist.append(sample)
        rv1d0 = multivariate_normal(mean[0], Sigma[0, 0])
        rv1d1 = multivariate_normal(mean[1], Sigma[1, 1])
        gaussian1dlist0.append(rv1d0)
        gaussian1dlist1.append(rv1d1)
        sample1dlist0.append(sample[:, 0])
        sample1dlist1.append(sample[:, 1])

    rvlist = [wght * rv.pdf(pos) for wght, rv in zip(weights, gaussianlist)]
    gmmpos = np.sum(rvlist, axis=0)

    rvlist1d0 = [wght * rv.pdf(testXplot0)
                 for wght, rv in zip(weights, gaussian1dlist0)]
    gmmpos1d0 = np.sum(rvlist1d0, axis=0)

    rvlist1d1 = [wght * rv.pdf(testXplot1)
                 for wght, rv in zip(weights, gaussian1dlist1)]
    gmmpos1d1 = np.sum(rvlist1d1, axis=0)

    fig = plt.figure(figsize=(10, 10))
    ax1 = plt.subplot2grid((5, 5), (0, 1), colspan=4, rowspan=4)
    ax1.set_xlim(xmin, xmax)
    ax1.set_ylim(ymin, ymax)
    ax2 = plt.subplot2grid((5, 5), (4, 1), colspan=4)
    ax2.set_xlim(xmin, xmax)
    ax3 = plt.subplot2grid((5, 5), (0, 0), rowspan=4)
    ax3.set_ylim(ymin, ymax)

    color = ['blue', 'red', 'green', 'purple', 'brown',
             'yellow', 'orange', 'white', 'black', 'gray']

    for i in range(n):
        ax1.scatter(samplelist[i][:, 0], samplelist[i][:, 1], s=10.,
                    alpha=0.5, color=color[i],
                    label="Samples from Rock Unit {}".format(i + 1))
        ax2.plot(testXplot0, weights[i] * gaussian1dlist0[i].pdf(testXplot0),
                 color=color[i], linewidth=3.,
                 label="Gaussian Dist.\nfor Rock Unit {}".format(i + 1))
        ax3.plot(weights[i] * gaussian1dlist1[i].pdf(testXplot1), testXplot1,
                 color=color[i], linewidth=3.)
    ax1.legend()
    ax2.plot(testXplot0, gmmpos1d0, color='k', label='Mixture\nDistribution')
    ax2.legend(loc=1)
    ax2.hist(sample1dlist0, bins=100, stacked=True,
             density=True, color=color[:n])
    ax3.plot(gmmpos1d1, testXplot1, color='k')
    ax3.hist(sample1dlist1, bins=100, stacked=True, density=True,
             color=color[:n], orientation="horizontal")

    ax1.contour(x, y, gmmpos, 10, cmap=cm.viridis)
    ax2.set_xlabel('Property 1', fontsize=fontsize)
    ax3.set_ylabel('Property 2', fontsize=fontsize)
    plt.tight_layout()
    plt.show()


def GMM2D_TextBox_wrapper(
        n, mean0_str, mean1_str,
        scale0_str, scale1_str,
        rotation_str, weights_str):

    mean0 = np.array(list(map(float, mean0_str.split(sep=","))))
    mean1 = np.array(list(map(float, mean1_str.split(sep=","))))
    scale0 = np.array(list(map(float, scale0_str.split(sep=","))))
    scale1 = np.array(list(map(float, scale1_str.split(sep=","))))
    rotation = np.array(list(map(float, rotation_str.split(sep=","))))
    weights = np.array(list(map(float, weights_str.split(sep=","))))

    GMM2D_creation(n, mean0, mean1, scale0, scale1, rotation, nbsamples=1000,
                   weights=weights, dxy=1e3, fontsize=16, ncontour=10)


def GMM2D_TextBox_app():
    app = widgetify(
        GMM2D_TextBox_wrapper,
        manual=True,
        n=ipywidgets.BoundedIntText(value=2, min=1, max=10, step=1, description='# of clusters:',
                                    disabled=False),
        mean0_str=ipywidgets.Text(value="1.,0.", description='Means for Phys.Prop. 1',
                                  style={'description_width': 'initial'},
                                  layout=ipywidgets.Layout(width='50%')),
        mean1_str=ipywidgets.Text(value="0.,1.", description='Means for Phys.Prop. 2',
                                  style={'description_width': 'initial'},
                                  layout=ipywidgets.Layout(width='50%')),
        scale0_str=ipywidgets.Text(value="5.,1.", description='Standard Deviation for Phys.Prop. 1',
                                   style={'description_width': 'initial'},
                                   layout=ipywidgets.Layout(width='50%')),
        scale1_str=ipywidgets.Text(value="1.,5.", description='Standard Deviation for Phys.Prop. 2',
                                   style={'description_width': 'initial'},
                                   layout=ipywidgets.Layout(width='50%')),
        rotation_str=ipywidgets.Text(value="0.,45.", description='Rotation for each cluster',
                                     style={'description_width': 'initial'},
                                     layout=ipywidgets.Layout(width='50%')),
        weights_str=ipywidgets.Text(value="0.6,0.4", description='Weight for each cluster',
                                    style={'description_width': 'initial'},
                                    layout=ipywidgets.Layout(width='50%')),
    )

    return app


def GMM2D_Slider_wrapper(
    mean00,
    mean01,
    mean10,
    mean11,
    scale00,
    scale01,
    scale10,
    scale11,
    rotation0,
    rotation1,
    weights0
):

    mean0 = np.array([mean00, mean01])
    mean1 = np.array([mean10, mean11])
    scale0 = np.array([scale00, scale01])
    scale1 = np.array([scale10, scale11])
    rotation = np.array([rotation0, rotation1, ])
    weights = np.array([weights0, 1. - weights0])

    GMM2D_creation(2, mean0, mean1, scale0, scale1, rotation, nbsamples=1000,
                   weights=weights, dxy=1e3, fontsize=16, ncontour=10)


def GMM2D_Slider_app():
    app = widgetify(
        GMM2D_Slider_app,
        manual=False,
        mean00=ipywidgets.FloatSlider(
            value=-5,
            min=-10.,
            max=10.0,
            step=0.1,
            description='Mean Phys.Prop 1; Cluster 1',
            style={'description_width': 'initial'},
            layout=ipywidgets.Layout(width='50%')
        ),
        mean01=ipywidgets.FloatSlider(
            value=5,
            min=-10.,
            max=10.0,
            step=0.1,
            description='Mean Phys.Prop 1; Cluster 2',
            style={'description_width': 'initial'},
            layout=ipywidgets.Layout(width='50%')
        ),
        mean10=ipywidgets.FloatSlider(
            value=-5,
            min=-10.,
            max=10.0,
            step=0.1,
            description='Mean Phys.Prop 2; Cluster 1',
            style={'description_width': 'initial'},
            layout=ipywidgets.Layout(width='50%')
        ),
        mean11=ipywidgets.FloatSlider(
            value=5,
            min=-10.,
            max=10.0,
            step=0.1,
            description='Mean Phys.Prop 2; Cluster 2',
            style={'description_width': 'initial'},
            layout=ipywidgets.Layout(width='50%')
        ),
        scale00=ipywidgets.FloatSlider(
            value=1,
            min=0.1,
            max=10.0,
            step=0.1,
            description='Std-Dev Phys.Prop 1; Cluster 1',
            style={'description_width': 'initial'},
            layout=ipywidgets.Layout(width='50%')
        ),
        scale01=ipywidgets.FloatSlider(
            value=2,
            min=0.1,
            max=10.0,
            step=0.1,
            description='Std-Dev Phys.Prop 1; Cluster 2',
            style={'description_width': 'initial'},
            layout=ipywidgets.Layout(width='50%')
        ),
        scale10=ipywidgets.FloatSlider(
            value=2,
            min=0.1,
            max=10.0,
            step=0.1,
            description='Std-Dev Phys.Prop 2; Cluster 1',
            style={'description_width': 'initial'},
            layout=ipywidgets.Layout(width='50%')
        ),
        scale11=ipywidgets.FloatSlider(
            value=1,
            min=0.1,
            max=10.0,
            step=0.1,
            description='Std-Dev Phys.Prop 2; Cluster 2',
            style={'description_width': 'initial'},
            layout=ipywidgets.Layout(width='50%')
        ),
        rotation0=ipywidgets.FloatSlider(
            value=0,
            min=0,
            max=180.0,
            step=1,
            description='Rotation angle cluster 1',
            style={'description_width': 'initial'},
            layout=ipywidgets.Layout(width='50%')
        ),
        rotation1=ipywidgets.FloatSlider(
            value=0,
            min=0,
            max=180.0,
            step=1,
            description='Rotation angle cluster 2',
            style={'description_width': 'initial'},
            layout=ipywidgets.Layout(width='50%')
        ),
        weights0=ipywidgets.FloatSlider(
            value=0.5,
            min=0,
            max=1,
            step=0.1,
            description='Weight cluster 1',
            style={'description_width': 'initial'},
            layout=ipywidgets.Layout(width='50%')  # , height='80px')
        ),
    )

    return app
