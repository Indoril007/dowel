"""A `dowel.logger.LogOutput` for wanb.

It receives the input data stream from `dowel.logger`, then logs them in wandb
.

Note:
    Neither TensorboardX nor TensorBoard supports log parametric
    distributions. We add this feature by sampling data from a
    `tfp.distributions.Distribution` object.

"""
import functools
import warnings

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import os

from dowel import Histogram
from dowel import LoggerWarning
from dowel import LogOutput
from dowel import TabularInput
from dowel.utils import colorize

import wandb
wandb.login()


class WandbOutput(LogOutput):

    def __init__(self,
                 run=None,
                 persist=False,
                 **kwargs):

        if run is None:
            self.run = wandb.init(**kwargs)
        else:
            self.run = run

        self.persist = persist
        self._waiting_for_dump = []

    @property
    def types_accepted(self):
        """Pass these types to this logger output.

        The types in this tuple will be accepted by this output.

        :return: A tuple containing all valid input types.
        """
        return (TabularInput)

    def record(self, data, prefix=''):
        """Pass logger data to this output.

        :param data: The data to be logged by the output.
        :param prefix: A prefix placed before a log entry in text outputs.
        """
        if isinstance(data, TabularInput):
            self._waiting_for_dump.append(
                functools.partial(self._record_tabular, data=data))
        else:
            raise ValueError('Unacceptable type.')

    def dump(self, step=None):
        """Dump the contents of this output.

        :param step: The current run step.
        """
        # Log the tabular inputs, now that we have a step
        for p in self._waiting_for_dump:
            p(step=step)
            self._waiting_for_dump.clear()

    def close(self):
        """Close any files used by the output."""
        if not self.persist:
            self.run.finish()

    def __del__(self):
        """Clean up object upon deletion."""
        self.close()

    def _record_tabular(self, data, step):
        wandb.log(data.as_dict, step=step)


