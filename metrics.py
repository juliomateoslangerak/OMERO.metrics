"""These are some possibly useful code snippets"""

# Thresholding and labeling
from skimage.filters import threshold_otsu, apply_hysteresis_threshold, gaussian
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square, cube, octahedron, ball
from skimage.feature import peak_local_max
from scipy.spatial.distance import cdist
import numpy as np
from itertools import permutations
from functools import reduce


class Image:
    """
    This is the top level abstraction of an metrics image. It is supposed to manage all generalities concerning
    the raw data and metadata.
    """
    def __init__(self,
                 image: np.ndarray,
                 dimensions: tuple = ('z', 'c', 'x', 'y'),
                 pixel_size_um=None,
                 **kwargs):
        self.image = image
        self.dimensions = dimensions
        self.pixel_size_um = pixel_size_um


class PointSourceImage(Image):
    """
    An image containing one or any other type of point source
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.labels_image = np.zeros(self.image.shape, dtype=np.uint16)
        self.properties = list()
        self.positions = list()
        self.channel_permutations = list()
        self.distances = list()

    def _remove_outliers(self, input_labels, criteria):
        criteria = [(r.label, r.area) for r in regionprops(labels)]
        l = None
        v = 0
        for new_l, new_v in criteria:
            if new_v > v:
                v = new_v
                l = new_l
        labels[labels == l] = 0


class BeadsField2D(PointSourceImage):
    """
    Docs in here
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # TODO: reshape image if dimensions is not default
        # TODO: implement drift for a time dimension

    def segment_channel(self, channel, sigma, method='local_maxima', remove_center_spot=True):
        """Segment a given channel (3D numpy ndarray) to find the spots"""
        thresh = threshold_otsu(channel)

        gauss_filtered = np.copy(channel)
        gaussian(gauss_filtered, (1, 3, 3))

        if method == 'hysteresis':  # We may try here hysteresis threshold
            thresholded = apply_hysteresis_threshold(gauss_filtered, low=(thresh * .7), high=(thresh * 1.0))

        elif method == 'local_maxima':  # We are applying a local maxima algorithm
            peaks = peak_local_max(gauss_filtered,
                                   min_distance=sigma,
                                   threshold_abs=(thresh * .2),
                                   exclude_border=True,
                                   indices=False)
            thresholded = np.copy(gauss_filtered)
            thresholded[peaks] = thresholded.max()
            thresholded = apply_hysteresis_threshold(thresholded, low=(thresh * .9), high=(thresh * 1.0))

        bw = closing(thresholded, cube(sigma))
        cleared = clear_border(bw)

        labels = label(cleared)

        if remove_center_spot:
            areas = [(r.label, r.area) for r in regionprops(labels)]
            l = None
            a = 0
            for new_l, new_a in areas:
                if new_a > a:
                    a = new_a
                    l = new_l
            labels[labels == l] = 0
        return labels

    def segment_image(self):
        for c in range(self.image.shape[1]):
            self.labels_image[:, c] = self.segment_channel(self.image[:, c], sigma=30)

    def compute_channel_properties(self, channel, label_channel):
        """Analyzes and extracts the properties of a single channel"""

        ch_properties = list()

        regions = regionprops(label_channel, channel)

        for region in regions:
            ch_properties.append({'label': region.label,
                                  'area': region.area,
                                  'centroid': region.centroid,
                                  'weighted_centroid': region.weighted_centroid,
                                  'max_intensity': region.max_intensity,
                                  'mean_intensity': region.mean_intensity,
                                  'min_intensity': region.min_intensity
                                  })
        ch_positions = np.array([x['weighted_centroid'] for x in ch_properties])
        if self.pixel_size_um:
            ch_positions = ch_positions[0:] * self.pixel_size_um

        return ch_properties, ch_positions

    def compute_image_properties(self):
        for c in range(self.image.shape[1]):
            pr, pos = self.compute_channel_properties(self.image[:, c], self.labels_image[:, c])
            self.properties.append(pr)
            self.positions.append(pos)

    def compute_distances_matrix(self):
        """Calculates all possible distances between all channels and returns the interesting values"""

        if self.image.shape[0] == 1:
            raise Exception('Image has one only channel and no distances matrix can be calculated')
            return None
        if len(self.properties) == 0:  # We guess that properties have not been calculated yet
            self.compute_image_properties()

        n_dim = len(self.positions) - 1

        if n_dim < 2:
            raise Exception('Not enough dimensions to do a distance measurement')

        self.channel_permutations = list(permutations(range(len(self.positions)), 2))

        # Create a space to hold the distances. For every permutation we want to store:
        # Positions of a (indexes 0:2)
        # Distance to the closest spot in b (index 3)
        # Index of the nearest spot in b (index 4)

        for a, b in self.channel_permutations:
            # TODO: Try this
            distances_matrix = cdist(self.positions[a], self.positions[b])

            pairwise_distances = list()
            for p, d in zip(self.positions[a], distances_matrix):
                single_distance = list()
                single_distance.append(tuple(p))  # Adding the coordinates of spot in ch_a
                single_distance.append(d.min())  # Appending the 3D distance
                single_distance.append(d.argmin())  # Appending the index of the closest spot in ch_b
                pairwise_distances.append(single_distance)

            self.distances.append(pairwise_distances)
