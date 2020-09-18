from torch import nn
import torch


def drop_block(x, mask):
    return x * mask * mask.numel() / mask.sum()


class DropBlock2d(nn.Module):
    r"""Randomly zeroes 2D spatial blocks of the input tensor.
        As described in the paper
        `DropBlock: A regularization method for convolutional networks`_ ,
        dropping whole blocks of feature map allows to remove semantic
        information as compared to regular dropout.
        Args:
            p (float): probability of an element to be dropped.
            block_size (int): size of the block to drop
        Shape:
            - Input: `(N, C, H, W)`
            - Output: `(N, C, H, W)`
        .. _DropBlock: A regularization method for convolutional networks:
           https://arxiv.org/abs/1810.12890
        """

    def __init__(self, p=0.1, block_size=7):
        super(DropBlock2d, self).__init__()
        assert 0 <= p <= 1
        self.p = p
        self.block_size = block_size

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        _, _, h, w = x.size()
        gamma = self.get_gamma(h, w)
        mask = self.get_mask(x, gamma)
        y = drop_block(x, mask)
        return y

    @torch.no_grad()
    def get_mask(self, x, gamma):
        mask = torch.bernoulli(torch.ones_like(x.sum(dim=0, keepdim=True)) * gamma)
        mask = 1 - torch.max_pool2d(mask, kernel_size=self.block_size, stride=1, padding=self.block_size // 2)
        return mask

    def get_gamma(self, h, w):
        return self.p * (h * w) / (self.block_size ** 2) / \
               ((w - self.block_size + 1) * (h * self.block_size + 1))


class DropBlockScheduler(object):
    def __init__(self, model, batches: int, num_epochs: int, start_value=0.1, stop_value=1.):
        self.model = model
        self.iter = 0
        self.start_value = start_value
        self.num_iter = batches * num_epochs
        self.st_line = (stop_value - start_value) / self.num_iter
        self.groups = []
        self.value = start_value

        def coll_dbs(md):
            if hasattr(md, 'block_size'):
                self.groups.append(md)

        model.apply(coll_dbs)

    def update_values(self, value):
        for db in self.groups:
            db.p = value

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_value(self):
        self.value = self.st_line * self.iter + self.start_value

    def state_dict(self):
        return {
            key: value for key,
            value in self.__dict__.items() if (key != 'model' and key != 'groups')}

    def step(self):
        self.get_value()
        self.update_values(self.value)
        self.iter += 1
