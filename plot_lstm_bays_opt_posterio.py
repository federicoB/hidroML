import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def posterior(optimizer, memory_obs, sample_obs, y_obs, grid):
    X = np.stack((memory_obs,sample_obs)).reshape(-1,2)
    optimizer._gp.fit(X, y_obs)


    mu, sigma = optimizer._gp.predict(grid, return_std=True)
    return mu, sigma


def plot_gp(optimizer):
    x = np.stack((
        np.linspace(8, 256, 10000),
        np.linspace(8, 256, 10000)
    )
    ).reshape(-1,2)
    fig = plt.figure(figsize=(16, 10))
    steps = len(optimizer.space)
    fig.suptitle(
        'Gaussian Process and Utility Function After {} Steps'.format(steps),
        fontdict={'size': 30}
    )

    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
    axis = plt.subplot(gs[0],projection='3d')

    memory_obs = np.array([[res["params"]["memory"]] for res in optimizer.res])
    sample_obs = np.array([[res["params"]["sample_lenght"]] for res in optimizer.res])
    y_obs = np.array([res["target"] for res in optimizer.res])

    mu, sigma = posterior(optimizer, memory_obs,sample_obs, y_obs, x)
    axis.plot(memory_obs.flatten(), sample_obs.flatten(), y_obs, 'D', markersize=8, label=u'Observations', color='r')
    X = np.linspace(0,100,100)
    axis.plot_surface(X, X, mu.reshape(100,-1), '--', color='k', label='Prediction')

    plt.show()