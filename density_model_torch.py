import argparse
import torch

import models_torch as models
import utils


def inference(parameters, model_type):

    datum_l_cc = utils.read_images(parameters['image_path'], 'L-CC')
    datum_r_cc = utils.read_images(parameters['image_path'], 'R-CC')
    datum_l_mlo = utils.read_images(parameters['image_path'], 'L-MLO')
    datum_r_mlo = utils.read_images(parameters['image_path'], 'R-MLO')

    device = torch.device("cuda:0" if args.device_type == "gpu" else "cpu")

    if model_type == 'cnn':
        model = models.BaselineBreastModel(nodropout_probability=1.0, gaussian_noise_std=0.0).to(device)
        model.load_state_dict(torch.load(parameters["initial_parameters"]))
        x = {
            "L-CC": torch.Tensor(datum_l_cc).permute(0, 3, 1, 2).to(device),
            "L-MLO": torch.Tensor(datum_l_mlo).permute(0, 3, 1, 2).to(device),
            "R-CC": torch.Tensor(datum_r_cc).permute(0, 3, 1, 2).to(device),
            "R-MLO": torch.Tensor(datum_r_mlo).permute(0, 3, 1, 2).to(device),
        }
    elif model_type == 'histogram':
        model = models.BaselineHistogramModel(num_bins=parameters["bins_histogram"]).to(device)
        model.load_state_dict(torch.load(parameters["initial_parameters"]))
        x = torch.Tensor(utils.histogram_features_generator([
            datum_l_cc, datum_r_cc, datum_l_mlo, datum_r_mlo
        ], parameters)).to(device)
    else:
        raise RuntimeError(model_type)

    with torch.no_grad():
        prediction_density = model(x).numpy()

    print('Density prediction:\n'
          '\tAlmost entirely fatty (0):\t\t\t' + str(prediction_density[0, 0]) + '\n'
          '\tScattered areas of fibroglandular density (1):\t' + str(prediction_density[0, 1]) + '\n'
          '\tHeterogeneously dense (2):\t\t\t' + str(prediction_density[0, 2]) + '\n'
          '\tExtremely dense (3):\t\t\t\t' + str(prediction_density[0, 3]) + '\n')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run Inference')
    parser.add_argument('model_type')
    parser.add_argument('--bins-histogram', default=50)
    parser.add_argument('--initial-parameters', default=None)
    parser.add_argument('--device-type', default="cpu")
    parser.add_argument('--image-path', default="images/")
    args = parser.parse_args()

    parameters_ = {
        "bins_histogram": args.bins_histogram,
        "initial_parameters": args.initial_parameters,
        "device_type": args.device_type,
        "image_path": args.image_path,
    }

    if parameters_["initial_parameters"] is None:
        if args.model_type == "histogram":
            parameters_["initial_parameters"] = "saved_models/BreastDensity_BaselineHistogramModel/pytorch_model.p"
        if args.model_type == "cnn":
            parameters_["initial_parameters"] = "saved_models/BreastDensity_BaselineBreastModel/pytorch_model.p"

    inference(parameters_, args.model_type)

"""
python density_model_torch.py histogram
python density_model_torch.py cnn
"""
