# Copyright (c) ModelScope Contributors. All rights reserved.
# code borrowed from https://github.com/jiaweizzhao/GaLore

import torch


class GaLoreOptimizerStateMixin:
    """Serialize projector state as tensors and primitives for weights-only checkpoint loading."""

    def state_dict(self):
        state_dict = super().state_dict()
        # Copy entries before substituting snapshots: Optimizer exposes live state dictionaries.
        state_dict['state'] = {
            pid: {**state, 'projector': state['projector'].state_dict()}
            if isinstance(state.get('projector'), GaLoreProjector) else state
            for pid, state in state_dict['state'].items()
        }
        return state_dict

    def load_state_dict(self, state_dict, *args, **kwargs):
        state = {}
        for pid, entry in state_dict['state'].items():
            projector = entry.get('projector')
            if isinstance(projector, dict) and projector.get('__galore_projector__'):
                entry = {**entry, 'projector': GaLoreProjector.from_state_dict(projector)}
            state[pid] = entry
        # Rebuild before Optimizer recursively casts dictionaries to the parameter dtype.
        super().load_state_dict({**state_dict, 'state': state}, *args, **kwargs)
        # bnb's Optimizer2State.load_state_dict has a real move_to_device flag: with False the
        # caller wants state kept on its SAVED device, so the projector must not be moved either.
        if not (args[0] if args else kwargs.get('move_to_device', True)):
            return
        for param, entry in self.state.items():
            projector = entry.get('projector')
            if isinstance(projector, GaLoreProjector) and projector.ortho_matrix is not None:
                # Keep the optimizer's computation dtype, which may differ from the parameter dtype.
                matrix = projector.ortho_matrix
                projector.ortho_matrix = ([tensor.to(device=param.device) for tensor in matrix]
                                          if isinstance(matrix, (list, tuple)) else matrix.to(device=param.device))


class GaLoreProjector:

    def __init__(self, rank, verbose=False, update_proj_gap=200, scale=1.0, proj_type='std'):
        self.rank = rank
        self.verbose = verbose
        self.update_proj_gap = update_proj_gap
        self.scale = scale
        self.ortho_matrix = None
        self.proj_type = proj_type

    def state_dict(self):
        """A plain-container snapshot, including the projection's computation dtype."""
        return {'__galore_projector__': True, 'rank': self.rank, 'verbose': self.verbose,
                'update_proj_gap': self.update_proj_gap, 'scale': self.scale,
                'proj_type': self.proj_type, 'ortho_matrix': self.ortho_matrix}

    @classmethod
    def from_state_dict(cls, state):
        projector = cls(state['rank'], verbose=state['verbose'], update_proj_gap=state['update_proj_gap'],
                        scale=state['scale'], proj_type=state['proj_type'])
        projector.ortho_matrix = state['ortho_matrix']
        return projector

    def project(self, full_rank_grad, iter):

        if self.proj_type == 'std':
            if full_rank_grad.shape[0] >= full_rank_grad.shape[1]:
                if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                    self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='right')
                low_rank_grad = torch.matmul(full_rank_grad, self.ortho_matrix.t())
            else:
                if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                    self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='left')
                low_rank_grad = torch.matmul(self.ortho_matrix.t(), full_rank_grad)
        elif self.proj_type == 'reverse_std':
            if full_rank_grad.shape[0] >= full_rank_grad.shape[1]:
                if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                    self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='left')
                low_rank_grad = torch.matmul(self.ortho_matrix.t(), full_rank_grad)
            else:
                if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                    self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='right')
                low_rank_grad = torch.matmul(full_rank_grad, self.ortho_matrix.t())
        elif self.proj_type == 'right':
            if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='right')
            low_rank_grad = torch.matmul(full_rank_grad, self.ortho_matrix.t())
        elif self.proj_type == 'left':
            if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='left')
            low_rank_grad = torch.matmul(self.ortho_matrix.t(), full_rank_grad)
        elif self.proj_type == 'full':
            if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='full')
            low_rank_grad = torch.matmul(self.ortho_matrix[0].t(), full_rank_grad) @ self.ortho_matrix[1].t()

        return low_rank_grad

    def project_back(self, low_rank_grad):

        if self.proj_type == 'std':
            if low_rank_grad.shape[0] >= low_rank_grad.shape[1]:
                full_rank_grad = torch.matmul(low_rank_grad, self.ortho_matrix)
            else:
                full_rank_grad = torch.matmul(self.ortho_matrix, low_rank_grad)
        elif self.proj_type == 'reverse_std':
            if low_rank_grad.shape[0] <= low_rank_grad.shape[1]:  # note this is different from std
                full_rank_grad = torch.matmul(self.ortho_matrix, low_rank_grad)
            else:
                full_rank_grad = torch.matmul(low_rank_grad, self.ortho_matrix)
        elif self.proj_type == 'right':
            full_rank_grad = torch.matmul(low_rank_grad, self.ortho_matrix)
        elif self.proj_type == 'left':
            full_rank_grad = torch.matmul(self.ortho_matrix, low_rank_grad)
        elif self.proj_type == 'full':
            full_rank_grad = torch.matmul(self.ortho_matrix[0], low_rank_grad) @ self.ortho_matrix[1]

        return full_rank_grad * self.scale

    # svd decomposition
    def get_orthogonal_matrix(self, weights, rank, type):
        module_params = weights

        if module_params.data.dtype != torch.float:
            float_data = False
            original_type = module_params.data.dtype
            original_device = module_params.data.device
            matrix = module_params.data.float()
        else:
            float_data = True
            matrix = module_params.data

        U, s, Vh = torch.linalg.svd(matrix, full_matrices=False)

        # make the smaller matrix always to be orthogonal matrix
        if type == 'right':
            A = U[:, :rank] @ torch.diag(s[:rank])
            B = Vh[:rank, :]

            if not float_data:
                B = B.to(original_device).type(original_type)
            return B
        elif type == 'left':
            A = U[:, :rank]
            B = torch.diag(s[:rank]) @ Vh[:rank, :]
            if not float_data:
                A = A.to(original_device).type(original_type)
            return A
        elif type == 'full':
            A = U[:, :rank]
            B = Vh[:rank, :]
            if not float_data:
                A = A.to(original_device).type(original_type)
                B = B.to(original_device).type(original_type)
            return [A, B]
        else:
            raise ValueError('type should be left, right or full')
