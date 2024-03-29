# -*- coding: utf-8 -*-
import logging
from tensorboardX import SummaryWriter


class TensorboardWriter(SummaryWriter):
    def __init__(self, logdir):
        """
        Extended SummaryWriter Class from tensorboard-pytorch (tensorbaordX)
        https://github.com/lanpa/tensorboard-pytorch/blob/master/tensorboardX/writer.py
        Internally calls self.file_writer
        """
        super(TensorboardWriter, self).__init__(logdir)
        self.logdir = self.file_writer.get_logdir()

    def update_parameters(self, module, step_i):
        """
        module: nn.Module
        """
        for name, param in module.named_parameters():
            self.add_histogram(name, param.clone().cpu().data.numpy(), step_i)

    def update_loss(self, loss, step_i, name='loss'):
        self.add_scalar(name, loss, step_i)

    def update_histogram(self, values, step_i, name='hist'):
        self.add_histogram(name, values, step_i)


def get_logger(log_dir, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter('[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s'
                                  )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(log_dir, 'a')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger