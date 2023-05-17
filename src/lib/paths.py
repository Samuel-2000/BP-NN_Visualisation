"""
creates folders, and simplifies working with file paths.

Samuel Kuchta <xkucht11@stud.fit.vutbr.cz> (2023)
inspired by: https://github.com/suzrz/vis_net_loss_landscape
"""
import os
from pathlib import Path
from .arg_parse import parse_arguments

args = parse_arguments()
model_arch = args.NNmodel
dataset_name = args.dataset
adam_true = args.adam
dropout = args.dropout

# general directories
current_model_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
dataset = Path(current_model_dir, "datasets")

models_dir = Path(current_model_dir, "models")

models_init_dir = Path(models_dir, "__models_initializations")
all_steps_dir = Path(models_dir, "_all_steps")


# current model directories
model_names = ["LeNet", "ModifiedLeNet", "VGG", "TinyNN", "TinyCNN"]
dataset_names = ["_MNIST", "_CIFAR-10", "_CIFAR-100"]

name = model_names[model_arch]
model_init_state = Path(models_init_dir, f"{name}_{dataset_names[dataset_name]}_init_state.pt")  # init_state
name += "_Adam" if adam_true else "_SGD"
name += "_dropout" if dropout == 1 else ""
name += dataset_names[dataset_name]


current_model_dir = Path(models_dir, name)

results = Path(current_model_dir, "results")
imgs = Path(current_model_dir, "images")


# model states
checkpoints = Path(current_model_dir, "model_states")
experiment_checkpoints = Path(current_model_dir, "experiment_model_states")
final_state = Path(checkpoints, "final_state.pt")
final_experiment_state = Path(experiment_checkpoints, "final_state.pt")

# directories for loss and accuracy during training
loss_acc = Path(results, "Training_results")

train_loss_path = Path(loss_acc, "train_loss_path.txt")
experiment_train_loss_path = Path(loss_acc, "step_size_experiment_train_loss_path.txt")
validation_loss_path = Path(loss_acc, "validation_loss.txt")
validation_acc_path = Path(loss_acc, "validation_accuracy.txt")


# directories for interpolation experiments
interpolation = Path(results, "Interpolation")
interpolation_img = Path(imgs, "Interpolation")

layers = Path(interpolation, "Layers")

loss_interpolated_model_path = Path(interpolation, "loss_interpolated_model.txt")
accuracy_interpolated_model_path = Path(interpolation, "accuracy_interpolated_model.txt")

loss_interpolated_layer_path = Path(layers, "loss_interpolated_layer")
accuracy_interpolated_layer_path = Path(layers, "accuracy_interpolated_layer")


# directory for steps
step_res = Path(results, "Steps")
step_img = Path(imgs, "Steps")

# directory for PCA directions
Surface = Path(results, "Surface")
Surface_img = Path(imgs, "Surface")


def init_dirs():
    """
    Function initializes directories
    """
    dirs = [results, imgs, loss_acc, checkpoints, models_init_dir, ]
    if args.linear:
        dirs.extend([interpolation, interpolation_img, layers])
    if args.step:
        dirs.extend([step_res, step_img, all_steps_dir])
    if args.step_experiment:
        dirs.append(experiment_checkpoints)

    if args.surface_all or args.surface_pca or args.surface_avg:
        dirs.extend([Surface, Surface_img])

    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
