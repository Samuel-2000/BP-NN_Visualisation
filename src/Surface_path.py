"""
Neural Network Training Progress Path Visualization using PCA directions

Samuel Kuchta <xkucht11@stud.fit.vutbr.cz> (2023)
"""
import prep
from lib import data_loader, examine_surface_path


def run_surface(args, device):
    """
    Runs the visualization of loss surface around trained model.

    :param args: experiment configuration
    :param device: device to be used
    """
    train_loader, test_loader, pca_train_loader = data_loader.data_load(args)
    model = prep.get_net(device, train_loader, test_loader, args)
    examine_surface_path.Examinator2D(args, model, device).get_loss_grid(pca_train_loader)
