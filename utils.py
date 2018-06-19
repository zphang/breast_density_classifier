import numpy as np
from scipy import misc


def histogram_features_generator(image_batch, parameters):

    def histogram_generator(img, bins):

        hist = np.histogram(img, bins=bins, density=False)
        hist_result = hist[0] / (hist[0].sum())

        return hist_result

    histogram_features = []
    x = [image_batch[0], image_batch[1], image_batch[2], image_batch[3]]

    for view in x:
        hist_img = []

        for i in range(view.shape[0]):
            hist_img.append(histogram_generator(view[i], parameters['bins_histogram']))

        histogram_features.append(np.array(hist_img))

    histogram_features = np.concatenate(histogram_features, axis=1)

    return histogram_features


def read_images(image_path, view):

    def normalise_single_image(image_):
        image_ -= np.mean(image_)
        image_ /= np.std(image_)

    image = misc.imread(image_path + view + '.png')
    image = image.astype(np.float32)
    normalise_single_image(image)
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=3)

    return image
