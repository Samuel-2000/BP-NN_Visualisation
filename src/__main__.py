"""
Main execute script

Samuel Kuchta <xkucht11@stud.fit.vutbr.cz> (2023)
"""
import sys
import os

# setting of workspace. change based on your filesystem location.
# might not be necessary if using IDE instead of command line.
if os.name == 'posix':
    # Google Colab (Linux)
    sys.path.insert(0, '/content/BP_proj/src')
else:
    # Windows
    sys.path.insert(0, 'C:\\Users\\Samuel\\Desktop\\BP_proj\\BP-NN_Visualisation\\src')

import sys
import torch
import Surface_path
import step_sizes
import linear_interpolation
import prep
from lib import paths
from lib.arg_parse import parse_arguments

if __name__ == '__main__':
    args = parse_arguments()
    paths.init_dirs()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using: {device}")

    if args.train:
        print(f"\tin train")
        prep.train_net(args, device)

    # visualisations
    if args.linear:
        print(f"\tin linear")
        linear_interpolation.linear(args, device)

    if args.step:
        print(f"\tin steps")
        step_sizes.compute_steps(args, device)

    if args.surface_all or args.surface_pca or args.surface_avg:
        print(f"\tin surface")
        Surface_path.run_surface(args, device)

sys.exit(0)
