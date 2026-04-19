#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import Optional, List

class LoRALayer():
    def __init__(
        self, 
        r: int, 
        lora_alpha: int, 
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights


class Embedding(nn.Embedding, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        r: int = 0,
        lora_alpha: int = 1,
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Embedding.__init__(self, num_embeddings, embedding_dim, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=0,
                           merge_weights=merge_weights)
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, num_embeddings)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((embedding_dim, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()

    def reset_parameters(self):
        nn.Embedding.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.zeros_(self.lora_A)
            nn.init.normal_(self.lora_B)

    def train(self, mode: bool = True):
        nn.Embedding.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    self.weight.data -= (self.lora_B @ self.lora_A).transpose(0, 1) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    self.weight.data += (self.lora_B @ self.lora_A).transpose(0, 1) * self.scaling
                self.merged = True
        
    def forward(self, x: torch.Tensor):
        if self.r > 0 and not self.merged:
            result = nn.Embedding.forward(self, x)
            after_A = F.embedding(
                x, self.lora_A.transpose(0, 1), self.padding_idx, self.max_norm,
                self.norm_type, self.scale_grad_by_freq, self.sparse
            )
            result += (after_A @ self.lora_B.transpose(0, 1)) * self.scaling
            return result
        else:
            return nn.Embedding.forward(self, x)
            

class Linear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        #print('reinstall is successed')
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize B the same way as the default for nn.Linear and A to zero
            # this is different than what is described in the paper but should not affect performance
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    self.weight.data -= T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    self.weight.data += T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = True       

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)            
            result += (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)


class SSVDLinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        off_diag: int = 0,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=0.0, merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        self.svd_initialized = False
        self.in_features = in_features
        self.out_features = out_features

        self.r_svft = min(out_features, in_features)
        self.k_trainable = int(self.r_svft // lora_dropout)

        # Non-trainable SVD components
        if self.out_features >= self.in_features:
            self.register_buffer("u", torch.empty(out_features, self.r_svft))
            self.register_buffer("v", torch.empty(self.r_svft, in_features))
        else:
            self.register_buffer("u", torch.empty(in_features, self.r_svft))
            self.register_buffer("v", torch.empty(self.r_svft, out_features))

        self.register_buffer("s_pre", torch.empty(self.r_svft))
        self.s = nn.Parameter(torch.zeros(self.k_trainable))

        self.gate = nn.Parameter(torch.empty(1).zero_(), requires_grad=True)

        self.K_vec = nn.Parameter(torch.zeros((self.k_trainable * (self.k_trainable - 1)) // 2), requires_grad=True)
        self.register_buffer('K_triu_idx', torch.triu_indices(self.k_trainable, self.k_trainable, offset=1))

        self.weight.requires_grad = False
        self.reset_parameters()

    def apply_svd(self):
        if not self.svd_initialized:
            if self.out_features >= self.in_features:
                u, s, v = torch.linalg.svd(self.weight, full_matrices=False)
            else:
                u, s, v = torch.linalg.svd(self.weight.T, full_matrices=False)
            self.u.data = u.detach()
            self.v.data = v.detach()
            self.s_pre.data = s.detach()
            self.gate.data = torch.tensor([0.], device=s.device)
            nn.init.kaiming_uniform_(self.s[None, :])
            self.s.squeeze()
            self.svd_initialized = True

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)

    def get_sigma(self):
        delta = F.pad(self.s * F.sigmoid(self.gate), (0, self.r_svft - self.k_trainable))
        return self.s_pre + delta

    def apply_rotation(self):
        Y = self.v.clone()
        k = self.k_trainable
        idx = self.K_triu_idx
        device = Y.device
        dtype = Y.dtype

        # Build A
        A = torch.eye(k, device=device, dtype=dtype)
        A[idx[0], idx[1]] -= 2 * self.K_vec
        A[idx[1], idx[0]] += 2 * self.K_vec

        # Apply A to top-k
        Y_top = Y[:k, :]
        Y_rest = Y[k:, :]

        Y_top_new = A @ Y_top
        Y_new = torch.cat([Y_top_new, Y_rest], dim=0)
        return Y_new

    def train(self, mode: bool = True):
        def T(w): return w.T if self.fan_in_fan_out else w
        
        nn.Linear.train(self, mode)

        if mode:
            if self.merge_weights and self.merged:
                self.weight.data = self.weight.data
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                sigma = self.get_sigma()
                if self.out_features >= self.in_features:
                    self.weight.data = T(self.u @ torch.diag(sigma) @ self.apply_rotation())
                else:
                    self.weight.data = T((self.u @ torch.diag(sigma) @ self.apply_rotation()).T)
                self.merged = True

    def forward(self, x: torch.Tensor):
        def T(w): return w.T if self.fan_in_fan_out else w
        self.apply_svd()
        sigma = self.get_sigma()
        if not self.merged:
            if self.out_features >= self.in_features:
                return F.linear(x, T(self.u @ torch.diag(sigma) @ self.apply_rotation()), bias=self.bias)
            else:
                return F.linear(x, T((self.u @ torch.diag(sigma) @ self.apply_rotation()).T), bias=self.bias)
        else:
            return F.linear(x, T(self.weight), bias=self.bias)
        
class DoraLinear(nn.Linear, LoRALayer):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)
        self.m_initialized = False

        self.weight_m_wdecomp = nn.Parameter(torch.ones((out_features, 1))) 

        self.fan_in_fan_out = fan_in_fan_out
        self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
        self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
        self.scaling = self.lora_alpha / self.r

        self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)
        
    def apply_m(self):

        if not self.m_initialized: 
            self.weight_m_wdecomp.data = torch.linalg.norm(self.weight, dim=1).unsqueeze(1).detach()
            self.m_initialized = True    

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
    
    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)

        if mode:
            if self.merge_weights and self.merged:
                self.weight.data = self.weight.data
                self.merged = False
        else:
            if self.merge_weights and not self.merged:        
                new_weight_v = (self.weight + (self.lora_B @ self.lora_A) * self.scaling) 
                norm_scale = self.weight_m_wdecomp.transpose(0, 1).view(-1) / (torch.linalg.norm(new_weight_v,dim=1)).detach()
                self.weight.data = self.weight + ((norm_scale-1) * self.weight.T + norm_scale * (self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling).T
                self.merged = True

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        
        self.apply_m()
        
        if self.r > 0 and not self.merged:
            new_weight_v = (self.weight + (self.lora_B @ self.lora_A) * self.scaling)
            norm_scale = self.weight_m_wdecomp.transpose(0, 1).view(-1) / (torch.linalg.norm(new_weight_v,dim=1)).detach()
            org_result = (F.linear(x, T(self.weight), bias=self.bias))
            dropout_x = self.lora_dropout(x)
            result = org_result + ((norm_scale-1) * (F.linear(dropout_x, T(self.weight)))).to(org_result.dtype)
            result += (norm_scale * (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1))) * self.scaling

        else:
            result = F.linear(x, T(self.weight), bias=self.bias)

        return result

class SVFTLinear(nn.Linear, LoRALayer):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        off_diag: int = 0,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=0, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)
        self.fan_in_fan_out = fan_in_fan_out
        self.svd_initialized = False

        self.r_svft = min(out_features, in_features)

        self.off_diag = r

        # === Collect banded (i, j) indices ===
        row_idx = []
        col_idx = []
        for d in range(-self.off_diag, self.off_diag + 1):
            i = torch.arange(self.r_svft - abs(d))
            j = i + d
            if d < 0:
                i, j = j, i
            row_idx.append(i)
            col_idx.append(j)
        self.register_buffer("band_row", torch.cat(row_idx))
        self.register_buffer("band_col", torch.cat(col_idx))
        self.num_banded_params = len(self.band_row)

        # === Only train the banded values ===
        self.m_entries = nn.Parameter(torch.zeros(self.num_banded_params))

        self.register_buffer("u", torch.empty(out_features, self.r_svft))
        self.register_buffer("v", torch.empty(self.r_svft, in_features))
        self.register_buffer("s_pre", torch.empty(self.r_svft))
        self.gate = nn.Parameter(torch.tensor(0.0), requires_grad=True)

        self.weight.requires_grad = False
        self.reset_parameters()

    def apply_svd(self):
        """Applies SVD to the current weight matrix."""

        if not self.svd_initialized: #or self.previous_weight_hash != current_weight_hash:
            u, s, v = torch.linalg.svd(self.weight, full_matrices=False)
            device = s.device

            self.u.data = u.clone().detach().contiguous()
            self.v.data = v.clone().detach().contiguous()
            self.s_pre.data = s.clone().detach().contiguous()
            self.svd_initialized = True
    
    def construct_M(self):
        # Place m_entries into correct (i, j) locations to form banded M
        M = torch.zeros(self.r_svft, self.r_svft, device=self.m_entries.device)
        M[self.band_row, self.band_col] = self.m_entries
        return M

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)

        if mode:
            if self.merge_weights and self.merged:
                self.weight.data = self.weight.data
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                M = self.construct_M() * torch.sigmoid(self.gate)
                self.weight.data = T(self.u @ (torch.diag(self.s_pre) + M) @ self.v)
                self.merged = True

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        self.apply_svd()

        if not self.merged:  
            M = self.construct_M() * torch.sigmoid(self.gate)
            result = F.linear(x, T(self.u @ (torch.diag(self.s_pre) + M) @ self.v), bias=self.bias)
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)

class PiSSALinear(nn.Linear, LoRALayer):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)
        self.fan_in_fan_out = fan_in_fan_out
        self.pissa_init = True
        self.pissa_con = True

        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)), requires_grad=True)
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)), requires_grad=True)
            self.scaling = self.lora_alpha / self.r
            self.weight.requires_grad = False

        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

        self.register_buffer("A0", torch.empty(r, in_features))
        self.register_buffer("B0", torch.empty(out_features, r))

        # rank-2r LoRA-delta buffers, filled on convert()
        self.register_buffer("delta_A", torch.empty(2 * r, in_features))
        self.register_buffer("delta_B", torch.empty(out_features, 2 * r))
        self.register_buffer("is_converted", torch.tensor(0, dtype=torch.uint8))

        self._pissa_factorize()

    def _pissa_factorize(self):
        if self.pissa_init:
            def T(w):
                return w.transpose(0, 1) if self.fan_in_fan_out else w

            U, S, Vh = torch.linalg.svd(self.weight, full_matrices=False)
            U_r, S_r, V_r = U[:, :self.r], S[:self.r], Vh[:self.r, :].T

            sqrtS = torch.sqrt(S_r)

            self.lora_A.data = (sqrtS.unsqueeze(1) * V_r.T)  # (r, d_in)
            self.lora_B.data = U_r * sqrtS.unsqueeze(0)      # (d_out, r)

            self.A0.data = (sqrtS.unsqueeze(1) * V_r.T)  # (r, d_in)
            self.B0.data = U_r * sqrtS.unsqueeze(0)      # (d_out, r)

            self.pissa_init = False

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, "lora_A"):
            nn.init.zeros_(self.lora_A)
            nn.init.zeros_(self.lora_B)
    
    
    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        A = torch.cat([self.lora_A, self.A0], dim=0)
        B = torch.cat([self.lora_B, -self.B0], dim=1)

        if mode:
            if self.merge_weights and self.merged:
                if self.r > 0:
                    self.weight.data -= T(B @ A) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                if self.r > 0:
                    self.weight.data += T(B @ A) * self.scaling
                self.merged = True
    

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        self._pissa_factorize()
        
        A = torch.cat([self.lora_A, self.A0], dim=0)
        B = torch.cat([self.lora_B, -self.B0], dim=1)

        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)
            result += (self.lora_dropout(x) @ A.transpose(0, 1) @ B.transpose(0, 1)) * self.scaling
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)
        
class MergedLinear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        enable_lora: List[bool] = [False],
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)
        assert out_features % len(enable_lora) == 0, \
            'The length of enable_lora must divide out_features'
        self.enable_lora = enable_lora
        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0 and any(enable_lora):
            self.lora_A = nn.Parameter(
                self.weight.new_zeros((r * sum(enable_lora), in_features)))
            self.lora_B = nn.Parameter(
                self.weight.new_zeros((out_features // len(enable_lora) * sum(enable_lora), r))
            ) # weights for Conv1D with groups=sum(enable_lora)
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
            # Compute the indices
            self.lora_ind = self.weight.new_zeros(
                (out_features, ), dtype=torch.bool
            ).view(len(enable_lora), -1)
            self.lora_ind[enable_lora, :] = True
            self.lora_ind = self.lora_ind.view(-1)
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def zero_pad(self, x):
        result = x.new_zeros((len(self.lora_ind), *x.shape[1:]))
        result[self.lora_ind] = x
        return result

    def merge_AB(self):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        delta_w = F.conv1d(
            self.lora_A.unsqueeze(0), 
            self.lora_B.unsqueeze(-1), 
            groups=sum(self.enable_lora)
        ).squeeze(0)
        return T(self.zero_pad(delta_w))

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0 and any(self.enable_lora):
                    self.weight.data -= self.merge_AB() * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0 and any(self.enable_lora):
                    self.weight.data += self.merge_AB() * self.scaling
                self.merged = True        

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        if self.merged:
            return F.linear(x, T(self.weight), bias=self.bias)
        else:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                result += self.lora_dropout(x) @ T(self.merge_AB().T) * self.scaling
            return result

class ConvLoRA(nn.Module, LoRALayer):
    def __init__(self, conv_module, in_channels, out_channels, kernel_size, r=0, lora_alpha=1, lora_dropout=0., merge_weights=True, **kwargs):
        super(ConvLoRA, self).__init__()
        self.conv = conv_module(in_channels, out_channels, kernel_size, **kwargs)
        for name, param in self.conv.named_parameters():
            self.register_parameter(name, param)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)
        assert isinstance(kernel_size, int)
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(
                self.conv.weight.new_zeros((r * kernel_size, in_channels * kernel_size))
            )
            self.lora_B = nn.Parameter(
              self.conv.weight.new_zeros((out_channels//self.conv.groups*kernel_size, r*kernel_size))
            )
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.conv.weight.requires_grad = False
        self.reset_parameters()
        self.merged = False

    def reset_parameters(self):
        self.conv.reset_parameters()
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode=True):
        super(ConvLoRA, self).train(mode)
        if mode:
            if self.merge_weights and self.merged:
                if self.r > 0:
                    # Make sure that the weights are not merged
                    self.conv.weight.data -= (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                if self.r > 0:
                    # Merge the weights and mark it
                    self.conv.weight.data += (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling
                self.merged = True

    def forward(self, x):
        if self.r > 0 and not self.merged:
            return self.conv._conv_forward(
                x, 
                self.conv.weight + (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling,
                self.conv.bias
            )
        return self.conv(x)

class Conv2d(ConvLoRA):
    def __init__(self, *args, **kwargs):
        super(Conv2d, self).__init__(nn.Conv2d, *args, **kwargs)

class Conv1d(ConvLoRA):
    def __init__(self, *args, **kwargs):
        super(Conv1d, self).__init__(nn.Conv1d, *args, **kwargs)

# Can Extend to other ones like this

class Conv3d(ConvLoRA):
    def __init__(self, *args, **kwargs):
        super(Conv3d, self).__init__(nn.Conv3d, *args, **kwargs)
