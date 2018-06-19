import argparse
import tensorflow as tf

import models_tf as models
import utils


def optimistic_restore(session, save_file):
    reader = tf.train.NewCheckpointReader(save_file)
    saved_shapes = reader.get_variable_to_shape_map()
    var_names = sorted(
        [(var.name, var.name.split(':')[0]) for var in tf.global_variables() if var.name.split(':')[0] in saved_shapes])
    restore_vars = []

    with tf.variable_scope('', reuse=True):

        for var_name, saved_var_name in var_names:
            curr_var = tf.get_variable(saved_var_name)
            var_shape = curr_var.get_shape().as_list()

            if var_shape == saved_shapes[saved_var_name]:
                restore_vars.append(curr_var)

    saver = tf.train.Saver(restore_vars)
    saver.restore(session, save_file)


def inference(parameters, model_type):
    tf.set_random_seed(7)

    with tf.device('/' + parameters['device_type']):
        if model_type == 'cnn':
            x_L_CC = tf.placeholder(tf.float32,
                                    shape=[None, parameters['input_size'][0], parameters['input_size'][1], 1])
            x_R_CC = tf.placeholder(tf.float32,
                                    shape=[None, parameters['input_size'][0], parameters['input_size'][1], 1])
            x_L_MLO = tf.placeholder(tf.float32,
                                     shape=[None, parameters['input_size'][0], parameters['input_size'][1], 1])
            x_R_MLO = tf.placeholder(tf.float32,
                                     shape=[None, parameters['input_size'][0], parameters['input_size'][1], 1])
            x = (x_L_CC, x_R_CC, x_L_MLO, x_R_MLO)
        elif model_type == 'histogram':
            x = tf.placeholder(tf.float32, shape=[None, parameters['bins_histogram'] * 4])

        nodropout_probability = tf.placeholder(tf.float32, shape=())
        Gaussian_noise_std = tf.placeholder(tf.float32, shape=())

        model = parameters['model_class'](parameters, x, nodropout_probability, Gaussian_noise_std)

        y_prediction_density = model.y_prediction_density

    if parameters['device_type'] == 'gpu':
        session_config = tf.ConfigProto()
        session_config.gpu_options.visible_device_list = str(parameters['gpu_number'])
    elif parameters['device_type'] == 'cpu':
        session_config = tf.ConfigProto(device_count={'GPU': 0})
    else:
        raise RuntimeError(parameters['device_type'])

    session = tf.Session(config=session_config)
    session.run(tf.global_variables_initializer())

    optimistic_restore(session, parameters['initial_parameters'])

    datum_L_CC = utils.read_images(parameters['image_path'], 'L-CC')
    datum_R_CC = utils.read_images(parameters['image_path'], 'R-CC')
    datum_L_MLO = utils.read_images(parameters['image_path'], 'L-MLO')
    datum_R_MLO = utils.read_images(parameters['image_path'], 'R-MLO')

    feed_dict_by_model = {nodropout_probability: 1.0, Gaussian_noise_std: 0.0}

    if model_type == 'cnn':
        feed_dict_by_model[x_L_CC] = datum_L_CC
        feed_dict_by_model[x_R_CC] = datum_R_CC
        feed_dict_by_model[x_L_MLO] = datum_L_MLO
        feed_dict_by_model[x_R_MLO] = datum_R_MLO
    elif model_type == 'histogram':
        feed_dict_by_model[x] = utils.histogram_features_generator(
            [datum_L_CC, datum_R_CC, datum_L_MLO, datum_R_MLO],
            parameters,
        )

    prediction_density = session.run(y_prediction_density, feed_dict=feed_dict_by_model)
    print('Density prediction:\n' +
          '\tAlmost entirely fatty (0):\t\t\t' + str(prediction_density[0, 0]) + '\n' +
          '\tScattered areas of fibroglandular density (1):\t' + str(prediction_density[0, 1]) + '\n' +
          '\tHeterogeneously dense (2):\t\t\t' + str(prediction_density[0, 2]) + '\n' +
          '\tExtremely dense (3):\t\t\t\t' + str(prediction_density[0, 3]) + '\n')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run Inference')
    parser.add_argument('model_type')
    parser.add_argument('--bins-histogram', default=50)
    parser.add_argument('--initial-parameters', default=None)
    parser.add_argument('--device-type', default="cpu")
    parser.add_argument('--gpu_number', default=0)
    parser.add_argument('--image-path', default="images/")
    args = parser.parse_args()

    parameters_ = {
        "bins_histogram": args.bins_histogram,
        "initial_parameters": args.initial_parameters,
        "device_type": args.device_type,
        "image_path": args.image_path,
        "gpu_number": args.gpu_number,
        "input_size": (2600, 2000),
    }

    if parameters_["initial_parameters"] is None:
        if args.model_type == "histogram":
            parameters_['model_class'] = models.BaselineHistogramModel
            parameters_["initial_parameters"] = "saved_models/BreastDensity_BaselineHistogramModel/model.ckpt"
        if args.model_type == "cnn":
            parameters_['model_class'] = models.BaselineBreastModel
            parameters_["initial_parameters"] = "saved_models/BreastDensity_BaselineBreastModel/model.ckpt"

    inference(parameters_, args.model_type)

"""
python density_model_tf.py histogram
python density_model_tf.py cnn
"""
