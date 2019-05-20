import matplotlib.pyplot as plt
from matplotlib import cm
import imageio
import metrics
import numpy as np
from scipy.interpolate import griddata

# raw_img = imageio.volread('/Users/julio/PycharmProjects/OMERO.metrics/Images/201702_RI508_Argolight-1-1_010_WF_ALX.ome.tif')
# raw_img = imageio.volread('/Users/julio/PycharmProjects/OMERO.metrics/Images/201702_RI508_Argolight-1-1_010_SIR_ALX.ome.tif')
raw_img = imageio.volread('/home/julio/PycharmProjects/OMERO.metrics/Images/Test_image_SIR_ALX.ome.tif')
# raw_img = imageio.volread('/Users/julio/Desktop/20170215_R506_Argolight_SIM_001_visit_13_WF.ome.tif')
# raw_img = imageio.volread('/Users/julio/Desktop/20160215_R506_Argolight_SIM_001_visit_13_SIR_ALX.dv/20160215_R506_Argolight_SIM_001_visit_13_SIR_ALX.ome.tif')
n_channels = raw_img.shape[1]
x_size = raw_img.shape[2]
y_size = raw_img.shape[3]

BF = metrics.BeadsField2D(raw_img)

BF.segment_image()
BF.compute_image_properties()
BF.compute_distances_matrix()


def plot_distances_maps(data, nb_of_channels, x_dim, y_dim):
    fig, axes = plt.subplots(ncols=(nb_of_channels - 1), nrows=nb_of_channels,
                             squeeze=False,
                             figsize=(9, 12),
                             constrained_layout=True
                             )
    ax = axes.ravel()

    for i, ch_pair in enumerate(data.channel_permutations):

        positions_map = np.asarray([p[0][1:] for p in data.distances[i]])
        distances_map = np.asarray([d[1] for d in data.distances[i]])
        grid_x, grid_y = np.mgrid[0:x_dim:1, 0:y_dim:1]
        interpolated = griddata(positions_map, distances_map, (grid_x, grid_y), method='cubic')

        ax[i].imshow(interpolated.T,
                                            extent=(0, x_dim, 0, y_dim),
                                            origin='lower',
                                            cmap=cm.hot,
                                            vmin=0,  # np.amin(raw_img[:, c, :, :]),
                                            vmax=10  # np.amax(raw_img[:, c, :, :])
                                            )
    plt.show()


def plot_homogeneity_map(data, nb_of_channels, x_dim, y_dim):

    fig, axes = plt.subplots(ncols=nb_of_channels, nrows=3, squeeze=False, figsize=(12, 6), constrained_layout=True)
    ax = axes.ravel()

    for c in range(nb_of_channels):
        # weighted_centroid = np.array([x['weighted_centroid'][0] for x in data.properties[c]])
        # areas = np.array([x['area'] for x in data.properties[c]])
        max_intensity = np.array([x['max_intensity'] for x in data.properties[c]])
        grid_x, grid_y = np.mgrid[0:x_dim:1, 0:y_dim:1]
        interpolated = griddata(data.positions[c][:, 1:], max_intensity, (grid_x, grid_y), method='cubic')

        # ax[c] = plt.subplot(3, 4, c + 1)

        ax[c].imshow(data.image[:, c, :, :].max(0), cmap='gray')
        ax[c].set_title('raw_channel_' + str(c))

        ax[c + nb_of_channels].imshow(data.labels_image[c].max(0))
        ax[c + nb_of_channels].set_title('segmented_channel_' + str(c))

        ax[c + 2 * nb_of_channels].imshow(interpolated.T,
                                          extent=(0, x_dim, 0, y_dim),
                                          origin='lower',
                                          cmap=cm.hot,
                                          vmin=np.amin(raw_img[:, c, :, :]),
                                          vmax=np.amax(raw_img[:, c, :, :]))
        ax[c + 2 * nb_of_channels].plot(data.positions[c][:, 1], data.positions[c][:, 2], 'k.', ms=2)
        # ax[c + 2 * nb_of_channels].clim(np.amin(raw_img[:, c, :, :]), np.amax(raw_img[:, c, :, :]))
        ax[c + 2 * nb_of_channels].set_title('Max_intensity_channel_' + str(c))

    plt.show()


plot_distances_maps(data=BF,
                    nb_of_channels=n_channels,
                    x_dim=x_size,
                    y_dim=y_size)

plot_homogeneity_map(data=BF,
                     nb_of_channels=n_channels,
                     x_dim=x_size,
                     y_dim=y_size)

# fig.colorbar(interpolated.T)

# out = metrics.analise_distances_matrix(positions)
