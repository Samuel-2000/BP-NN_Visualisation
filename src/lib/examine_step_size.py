"""
Neural Network Training Progress Steps Examinator

Samuel Kuchta <xkucht11@stud.fit.vutbr.cz> (2023)
"""
import os
import pickle
import numpy as np
from . import paths, functions
from pathlib import Path


class ExaminatorStepSize:
    def __init__(self, args, device):
        self.device = device
        if args.step_experiment:
            self.directory = paths.experiment_checkpoints
        else:
            self.directory = paths.checkpoints
        self.args = args

    def get_steps(self):
        """
        Calculate Steps

        :return: steps
        """
        if self.args.step_experiment:
            step_file = Path(os.path.join(paths.step_res, "experiment_steps"))
        else:
            step_file = Path(os.path.join(paths.step_res, "steps"))

        if step_file.exists():
            with open(step_file, "rb") as fd:
                steps = pickle.load(fd)
        else:
            steps = []

            prev_step = None
            files = os.listdir(os.path.abspath(self.directory))
            files.sort(key=functions.natural_keys)

            for filename in files:
                if "step" in filename:
                    with open(os.path.join(os.path.abspath(self.directory), filename), "rb") as fd:
                        checkpoint = pickle.load(fd)
                        step = checkpoint["flat_w"].cpu().numpy()

                        if prev_step is not None:
                            steps.append(np.sqrt(np.sum((step - prev_step) ** 2, axis=0)))

                        prev_step = step

            with open(step_file, "wb") as fd:
                pickle.dump(steps, fd)

        return steps
