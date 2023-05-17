"""
Neural Network Training Progress Steps Size Visualization

Samuel Kuchta <xkucht11@stud.fit.vutbr.cz> (2023)
"""
from lib import examine_step_size, plots


def compute_steps(args, device):
    """
    Runs the visualization of loss surface around trained model.

    :param args: program arguments
    :param device: device to be used
    """
    examine = examine_step_size.ExaminatorStepSize(args, device)
    steps = examine.get_steps()
    plots.plot_steps(args, steps)
    plots.plot_all_steps()
