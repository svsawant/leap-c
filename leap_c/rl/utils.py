""" This file contains utility functions that are used in the training loop. """

import torch
import torch.nn as nn


def soft_target_update(
    source_net: nn.Module, target_net: nn.Module, tau: float
) -> None:
    """Update the target network parameters using a soft update rule.

    Args:
        source_net: The source network whose parameters are copied to the target network.
        target_net: The target network whose parameters are updated.
        tau: The interpolation parameter for the soft update rule.
    """
    for source_param, target_param in zip(
        source_net.parameters(), target_net.parameters()
    ):
        target_param.data.copy_(
            tau * source_param.data + (1.0 - tau) * target_param.data
        )

class Normal(torch.distributions.Normal):
    '''Multivariate Gaussian distribution given mean and std tensors.'''

    def log_prob(self, actions):
        '''Log probability of actions given current distribution.

        Args:
            actions (torch.FloatTensor): shape (batch, *).

        Returns:
            (torch.FloatTensor): shape (batch, 1).
        '''
        return super().log_prob(actions).sum(-1, keepdim=True)

    def entropy(self):
        '''Entropy of current distribution.

        Returns:
            (torch.FloatTensor): shape (batch,).
        '''
        return super().entropy().sum(-1)

    def mode(self):
        '''Mode (max probability point) of current distribution.'''
        return self.mean