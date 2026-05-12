import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


LEAKY_ALPHA = 0.1
def init_param(modules):
    for m in modules:
        if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, a=LEAKY_ALPHA, mode='fan_out', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
            
            
class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
        super(TemporalConv, self).__init__()
        
        pad = (kernel_size + (kernel_size-1) * (dilation-1) - 1) // 2
        self.conv = nn.Conv2d(in_channels, 
                              out_channels, 
                              kernel_size=(kernel_size, 1),
                              padding=(pad, 0), 
                              stride=(stride, 1), 
                              dilation=(dilation, 1), 
                              groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x
    
    
class PointWiseTCN(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super(PointWiseTCN, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=(stride, 1), groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x
    
    
class UnfoldTemporalWindows(nn.Module):
    def __init__(self, window_size, window_stride, window_dilation=1, pad=True):
        super().__init__()
        
        self.window_size = window_size
        self.padding = (window_size + (window_size-1) * (window_dilation-1) - 1) // 2 if pad else 0
        
        self.unfold = nn.Unfold(kernel_size=(self.window_size, 1),
                                dilation=(self.window_dilation, 1),
                                stride=(self.window_stride, 1),
                                padding=(self.padding, 0))

    def forward(self, x):
        N, C, T, V = x.shape
        x = self.unfold(x)
        x = x.view(N, C, self.window_size, -1, V)
        x = x.transpose(2, 3).contiguous()
        return x
    
    
class PositionalEncoding(nn.Module):
    def __init__(self, channel, joint_num, time_len):
        super(PositionalEncoding, self).__init__()
        self.joint_num = joint_num
        self.time_len = time_len

        pos_list = []
        for t in range(self.time_len):
            for j_id in range(self.joint_num):
                pos_list.append(j_id)
        position = torch.from_numpy(np.array(pos_list)).unsqueeze(1).float()

        pe = torch.zeros(self.time_len * self.joint_num, channel)
        div_term = torch.exp(torch.arange(0, channel, 2).float() *
                             -(math.log(10000.0) / channel))  # channel//2
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.view(time_len, joint_num, channel).permute(2, 0, 1).unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):  # nctv
        x = x + self.pe.to(x.dtype)[:, :, :x.size(2)]
        return x   
    
    
class ST_GC(nn.Module):
    def __init__(self, in_channels, out_channels, A):
        super(ST_GC, self).__init__()
        
        A = torch.from_numpy(A.astype(np.float32))
        self.A = nn.Parameter(A)
        self.Nh = A.size(0)
        
        self.conv = nn.Conv2d(in_channels, out_channels * self.Nh, 1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        N, C, T, V = x.size()
        v = self.conv(x).view(N, self.Nh, -1, T, V)
        weights = self.A.to(v.dtype)
        x = torch.einsum('hvu,nhctu->nctv', weights, v)
        x = self.bn(x)
        return x
    

class CTR_GC(nn.Module):
    def __init__(self, in_channels, out_channels, A, num_scale=1):
        super(CTR_GC, self).__init__()

        A = torch.from_numpy(A.astype(np.float32))
        self.Nh = A.size(0)
        self.A = nn.Parameter(A)
        self.num_scale = num_scale
        
        rel_channels = in_channels // 8 if in_channels != 3 else 8
        
        self.conv1 = nn.Conv2d(in_channels, rel_channels * self.Nh, 1, groups=num_scale)
        self.conv2 = nn.Conv2d(in_channels, rel_channels * self.Nh, 1, groups=num_scale)
        self.conv3 = nn.Conv2d(in_channels, out_channels * self.Nh, 1, groups=num_scale)
        self.conv4 = nn.Conv2d(rel_channels * self.Nh, out_channels * self.Nh, 1, groups=num_scale * self.Nh)
        
        self.alpha = nn.Parameter(torch.zeros(1))
        self.bn = nn.BatchNorm2d(out_channels)
    
        self.tanh = nn.Tanh()
        self.relu = nn.LeakyReLU(LEAKY_ALPHA)

    def forward(self, x, A=None, alpha=1):
        N, C, T, V = x.size()
        res = x
        q, k, v = self.conv1(x).mean(-2), self.conv2(x).mean(-2), self.conv3(x).view(N, self.num_scale, self.Nh, -1, T, V)
        weights = self.conv4(self.tanh(q.unsqueeze(-1) - k.unsqueeze(-2))).view(N, self.num_scale, self.Nh, -1, V, V)        
        weights = weights * self.alpha.to(weights.dtype) + self.A.view(1, 1, self.Nh, 1, V, V).to(weights.dtype)
        x = torch.einsum('ngacvu, ngactu->ngctv', weights, v).contiguous().view(N, -1, T, V)
        x = self.bn(x)
        return x
    

class DeSGC(nn.Module):
    '''
    Note: This module is not included in the open-source release due to subsequent research and development. 
    It will be made available in future updates after the completion of related studies.
    '''
    def __init__(self, in_channels, out_channels, A, k, num_scale=4, num_frame=64, num_joint=25):
        super(DeSGC, self).__init__()

        A = torch.from_numpy(A.astype(np.float32))
        self.Nh = A.size(0)                 # head/subset 数
        self.A = nn.Parameter(A)            # a_ij: adjacency component in Eq.(2):contentReference[oaicite:2]{index=2}
        self.B = nn.Parameter(A.clone())    # b_ij: learnable adjacency in Eq.(5):contentReference[oaicite:3]{index=3}

        self.num_scale = num_scale
        self.k = k
        self.delta = 10                     # δ in Eq.(3):contentReference[oaicite:4]{index=4}

        rel_channels = in_channels // 8 if in_channels != 3 else 8
        self.factor = rel_channels // num_scale   # 每个 scale 的 rel 通道数

        self.pe = PositionalEncoding(in_channels, num_joint, num_frame)

        # θ(·): Cin -> Cout (但你这里一次性产生 Cout*Nh，并按 scale group 化)
        self.conv = PointWiseTCN(in_channels, out_channels * self.Nh, 1, groups=num_scale)

        # φ(·), ψ(·): Cin -> rel_channels (共享给 Eq.(2) 和 Eq.(5))
        self.convQK = nn.Conv2d(in_channels, 2 * rel_channels * self.Nh, 1, groups=num_scale)

        # ξ(·): rel_channels -> Cout (但这里输出 Cout*Nh)
        self.convW = nn.Conv2d(rel_channels * self.Nh, out_channels * self.Nh, 1, groups=num_scale * self.Nh)

        self.alpha = nn.Parameter(torch.zeros(1))                 # α in Eq.(2)
        self.beta  = nn.Parameter(torch.zeros(1, 1, self.Nh, 1, 1, 1))  # β in Eq.(5)（按 head 广播）
        self.bn = nn.BatchNorm2d(out_channels)

        self.tanh = nn.Tanh()
        self.relu = nn.LeakyReLU(LEAKY_ALPHA)

        # ---------------- vis cache (for Fig.8 style) ----------------
        self.enable_vis = False          # turn on/off
        self.vis_max_batch = 8           # cache at most N samples each forward
        self.vis_frame_idx = None        # None -> use T//2
        self._vis_cache = []             # list of dicts

    def forward(self, x):
        """
        x: (N, Cin, T, V)
        return: (N, Cout, T, V)
        """
        N, Cin, T, V = x.size()
        dtype, device = x.dtype, x.device

        # ------------------------------------------------------------
        # (0) positional encoding（实现细节论文没写死，你的类里已给出 pe）
        # ------------------------------------------------------------
        x_pe = x + self.pe(x)  # (N, Cin, T, V)

        # ------------------------------------------------------------
        # (1) θ(x): 用于 sampling 后的特征（Eq.(4) 的 θ(·) Cin->Cout）
        #     v: (N, num_scale, Nh, Cout_per_scale, T, V)
        # ------------------------------------------------------------
        v = self.relu(self.conv(x_pe))  # (N, Cout*Nh, T, V)
        Cout = v.size(1) // self.Nh
        assert Cout * self.Nh == v.size(1)
        assert Cout % self.num_scale == 0
        Cout_ps = Cout // self.num_scale  # 每个 scale 分到的输出通道
        v = v.view(N, self.num_scale, self.Nh, Cout_ps, T, V)

        # ------------------------------------------------------------
        # (2) φ(x), ψ(x) for Eq.(2) & Eq.(5)   (Fig.3(a) 顶部两分支):contentReference[oaicite:5]{index=5}
        #     convQK 是 2D conv：把 (T,V) 当作 2D 平面
        # ------------------------------------------------------------
        qk = self.convQK(x_pe)  # (N, 2*rel_channels*Nh, T, V)
        rel_channels = qk.size(1) // (2 * self.Nh)
        assert rel_channels % self.num_scale == 0
        rel_ps = rel_channels // self.num_scale

        q, k = torch.chunk(qk, 2, dim=1)  # each: (N, rel_channels*Nh, T, V)

        # reshape to multi-scale & multi-head
        q = q.view(N, self.num_scale, self.Nh, rel_ps, T, V)
        k = k.view(N, self.num_scale, self.Nh, rel_ps, T, V)

        # temporal average (1/T sum_t) in Eq.(2)/(5)
        q_bar = q.mean(dim=4)  # (N, S, H, rel_ps, V)
        k_bar = k.mean(dim=4)  # (N, S, H, rel_ps, V)

        # ------------------------------------------------------------
        # (3) calculate score π_ij (Eq.(2))
        #     π_ij = a_ij + α * σ( <φ(x_i), ψ(x_j)> )  (normalized inner product)
        # ------------------------------------------------------------
        qn = F.normalize(q_bar, p=2, dim=3)  # normalize over channel
        kn = F.normalize(k_bar, p=2, dim=3)

        # sim: (N, S, H, V, V)
        sim = torch.einsum('nshcv,nshcw->nshvw', qn, kn)
        sim = self.tanh(sim)  # σ(·) in Eq.(2), 用 tanh 稳定范围

        A = self.A.to(device=device, dtype=dtype).view(1, 1, self.Nh, V, V)  # a_ij
        pi = A + self.alpha.to(dtype) * sim  # (N, S, H, V, V)  —— Eq.(2):contentReference[oaicite:7]{index=7}

        # ------------------------------------------------------------
        # (4) Calibration Offset & k well-calibrated distributions (Eq.(3)):contentReference[oaicite:8]{index=8}
        #     对每个 center i：取 top-k 的索引 {p_i1,...,p_ik}
        #     对第 m 个分布：在对应 pim 位置加 δ，再 softmax -> bπ^(m)
        # ------------------------------------------------------------
        topk_val, topk_idx = torch.topk(pi, k=self.k, dim=-1)  # idx: (N,S,H,V,k)
        # ---------------- cache for visualization (Fig.8) ----------------
        if (not self.training) and self.enable_vis:
            with torch.no_grad():
                n_keep = min(N, int(self.vis_max_batch))

                # choose a frame to draw skeleton pose
                fidx = self.vis_frame_idx
                if fidx is None:
                    fidx = T // 2
                fidx = int(max(0, min(T - 1, fidx)))

                # cache topk indices: (N,S,H,V,k) -> keep first n_keep
                tk = topk_idx[:n_keep].detach().cpu()  # long tensor

                self._vis_cache.append({
                    "topk_idx": tk,        # (B,S,H,V,k)
                    "T": T,
                    "V": V,
                    "frame_idx": fidx,
                })

                # optional: keep cache small
                if len(self._vis_cache) > 20:
                    self._vis_cache = self._vis_cache[-20:]

        bpi_list = []
        for m in range(self.k):
            offset = torch.zeros_like(pi)  # (N,S,H,V,V)
            idx_m = topk_idx[..., m].unsqueeze(-1)  # (N,S,H,V,1)
            offset.scatter_(-1, idx_m, float(self.delta))  # 1_{bπ}=δ at pim else 0
            bpi_m = F.softmax(pi + offset, dim=-1)  # (N,S,H,V,V) —— Eq.(3)
            bpi_list.append(bpi_m)
        # (N,S,H,k,V,V)
        bpi = torch.stack(bpi_list, dim=3)

        # ------------------------------------------------------------
        # (5) Deformable spatial sampling (Eq.(4)):
        #     x~_c^(m) = Σ_j θ(x_j) * bπ_cj^(m)
        #     这里 θ(x_j) 就是 v (Cin->Cout 后的特征):contentReference[oaicite:9]{index=9}
        # ------------------------------------------------------------
        # sampled: (N,S,H,Cout_ps,T,V,k)
        sampled = torch.einsum('nshctj,nshmij->nshctim', v, bpi)

        # ------------------------------------------------------------
        # (6) calculate weight matrix w(x_i,x_j) (Eq.(5))
        #     w = b_ij + β * ξ( σ( mean_t( φ(x_i) - ψ(x_j) ) ) )
        #     Fig.3(a) 右支路是 ⊖ (pair-wise subtraction):contentReference[oaicite:10]{index=10}
        # ------------------------------------------------------------
        # diff: (N,S,H,rel_ps,V,V)
        diff = (q_bar.unsqueeze(-1) - k_bar.unsqueeze(-2))  # i-j subtraction
        diff = self.tanh(diff)

        # convW expects (N, rel_channels*Nh, V, V)
        diff2d = diff.reshape(N, self.num_scale, self.Nh, rel_ps, V, V)
        diff2d = diff2d.permute(0, 1, 2, 3, 4, 5).contiguous()
        diff2d = diff2d.view(N, self.Nh * (self.num_scale * rel_ps), V, V)  # = Nh*rel_channels

        w_dyn = self.convW(diff2d)  # (N, out_channels*Nh, V, V)
        w_dyn = w_dyn.view(N, self.num_scale, self.Nh, Cout_ps, V, V)

        B = self.B.to(device=device, dtype=dtype).view(1, 1, self.Nh, 1, V, V)  # b_ij
        w = B + self.beta.to(dtype) * w_dyn  # (N,S,H,Cout_ps,V,V) —— Eq.(5):contentReference[oaicite:11]{index=11}

        # ------------------------------------------------------------
        # (7) extract k weights using calibrated one-hot vectors (Eq.(6))
        #     ŵ_c,m = Σ_j w(c,j)*bπ_cj^(m)
        # ------------------------------------------------------------
        # bw: (N,S,H,Cout_ps,V,k)
        bw = torch.einsum('nshcij,nshmij->nshcim', w, bpi)  # Eq.(6):contentReference[oaicite:12]{index=12}

        # ------------------------------------------------------------
        # (8) spatial graph convolution (Eq.(7)):
        #     y_c = Σ_m ŵ_c,m * x~_c^(m)
        # ------------------------------------------------------------
        # sampled: (N,S,H,Cout_ps,T,V,k)
        bw2 = bw.unsqueeze(4)  # (N,S,H,Cout_ps,1,V,k)
        y = (sampled * bw2).sum(dim=-1)  # sum over k -> (N,S,H,Cout_ps,T,V)
        y = y.sum(dim=2)                  # sum over heads Nh (element-wise summation):contentReference[oaicite:13]{index=13}
        y = y.contiguous().view(N, Cout, T, V)  # merge scales back to Cout

        # BN over (T,V) as 2D plane
        y = self.bn(y)
        return y
    
    def plot_fig8_style(
        self,
        joint_i: int,
        skeleton_edges,
        raw_data,
        action_name="",
        sample_ids=(0, 1, 2, 3),
        cache_id: int = -1,
        scale_idx: int = 0,
        head_idx: int = 0,
        save_path: str = "fig8.png",
        figsize=(4.8, 5.2),
    ):
        """
        Use cached data to draw Fig.8-like visualization:
        - black: skeleton edges
        - green: top-k edges from joint_i to its selected neighbors
        - red dot: joint_i

        skeleton_edges: list[(u,v)] in joint index space
        sample_ids: which samples (within cached batch) to draw, length 4 recommended
        cache_id: which cache entry to use (-1 = latest)
        """
        import matplotlib.pyplot as plt
        assert len(self._vis_cache) > 0, "No vis cache found. Run a forward pass with enable_vis=True in eval mode."
        cache = self._vis_cache[cache_id]
        topk_idx = cache["topk_idx"]  # (B,S,H,V,k)

        # print(f"Using cache_id={cache_id} with raw_data shape {raw_data.shape} and topk_idx shape {topk_idx.shape}")
        B, _, _, V, _ = raw_data.shape
        joint_i = int(joint_i)
        assert 0 <= joint_i < V, f"joint_i out of range: {joint_i} vs V={V}"
        s = int(scale_idx)
        h = int(head_idx)

        # make 2x2 grid
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(int(math.sqrt(len(sample_ids))), int(math.sqrt(len(sample_ids))), wspace=0.15, hspace=0.15)

        sample_ids = list(sample_ids)

        for idx_plot, sid in enumerate(sample_ids):
            sid = int(sid)
            assert 0 <= sid < B, f"sample id {sid} out of cached batch range B={B}"
            ax = fig.add_subplot(gs[idx_plot // int(math.sqrt(len(sample_ids))), idx_plot % int(math.sqrt(len(sample_ids)))])
            ax.set_aspect("equal")
            ax.axis("off")

            raw_data_numpy = raw_data[sid].cpu().numpy()  # (C,T,V,M)
            fid = cache.get("frame_idx", raw_data_numpy.shape[1] // 2)  # default to middle frame
            xy = raw_data_numpy[:2, fid, :, 0]  # (2,V) take first person if multiple
            x = xy[0]
            y = -xy[1]  # flip y to look like image coordinates

            # draw skeleton edges (black)
            for (u, v) in skeleton_edges:
                u = int(u); v = int(v)
                if 0 <= u < V and 0 <= v < V:
                    ax.plot([x[u], x[v]], [y[u], y[v]], c=(0,0,0), lw=0.5)

            # pick top-k neighbors for this joint_i (use (s,h) slice)
            # topk_idx shape: (B,S,H,V,k) -> neighbors: (k,)
            neigh = topk_idx[sid, s, h, joint_i].numpy().tolist()
            # draw green edges from joint_i to each neighbor
            for j in neigh:
                j = int(j)
                if 0 <= j < V:
                    ax.plot([x[joint_i], x[j]], [y[joint_i], y[j]], linewidth=0.5, c=(0,1,0))

            # draw red dot at joint_i
            ax.scatter([x[joint_i]], [y[joint_i]], s=30)

        # dashed border + label (rough fig8 vibe)
        fig.suptitle(action_name, fontsize=16, y=0.02)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return save_path


class DeTGC(nn.Module):
    def __init__(self, in_channels, out_channels, eta, kernel_size=1, stride=1, padding=0, dilation=1, 
                 num_scale=1, num_frame=64):
        super(DeTGC, self).__init__()
        
        self.ks, self.stride, self.dilation = kernel_size, stride, dilation
        self.T = num_frame
        self.num_scale = num_scale
        
        self.eta = eta
        ref = (self.ks + (self.ks-1) * (self.dilation-1) - 1) // 2
        tr = torch.linspace(-ref, ref, self.eta)
        self.tr = nn.Parameter(tr)

        self.conv_out = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=(self.eta, 1, 1)),
            nn.BatchNorm3d(out_channels)
        )

    def forward(self, x):
        res = x
        N, C, T, V = x.size()
        Tout = T // self.stride
        dtype = x.dtype 
        
        #learnable sampling locations
        t0 = torch.arange(0, T, self.stride, dtype=dtype, device=x.device)
        tr = self.tr.to(dtype)
        t0, tr = t0.view(1, 1, -1).expand(-1, self.eta, -1), tr.view(1, self.eta, 1) 
        t = t0 + tr 
        t = t.view(1, 1, -1, 1) 
        
        #indexing
        tdn = t.detach().floor()
        tup = tdn + 1
        index1, index2 = torch.clamp(tdn, 0, self.T-1).long(), torch.clamp(tup, 0, self.T-1).long()
        index1, index2 = index1.expand(N, C, -1, V), index2.expand(N, C, -1, V)
        
        #sampling
        alpha = tup - t
        x1, x2 = x.gather(-2, index=index1), x.gather(-2, index=index2) 
        x = x1 * alpha + x2 * (1 - alpha)
        x = x.view(N, C, self.eta, Tout, V)
        
        #conv
        x = self.conv_out(x).squeeze(2)
        return x


class MultiScale_TemporalModeling(nn.Module):
    def __init__(self, in_channels, out_channels, eta, kernel_size=5, stride=1, dilations=1, 
                 num_scale=1, num_frame=64):
        super(MultiScale_TemporalModeling, self).__init__()
        
        scale_channels = out_channels // num_scale
        self.num_scale = num_scale if in_channels !=3 else 1

        self.tcn1 = nn.Sequential(
            PointWiseTCN(in_channels, scale_channels),
            nn.LeakyReLU(LEAKY_ALPHA),
            DeTGC(scale_channels, 
                  scale_channels, 
                  eta,
                  kernel_size=5, 
                  stride=stride, 
                  dilation=1, 
                  num_scale=num_scale, 
                  num_frame=num_frame)
        )
        
        self.tcn2 = nn.Sequential(
            PointWiseTCN(in_channels, scale_channels),
            nn.LeakyReLU(LEAKY_ALPHA),
            DeTGC(scale_channels, 
                  scale_channels, 
                  eta,
                  kernel_size=5, 
                  stride=stride, 
                  dilation=2, 
                  num_scale=num_scale, 
                  num_frame=num_frame)
        )
        
        self.maxpool3x1 = nn.Sequential(
            PointWiseTCN(in_channels, scale_channels),
            nn.LeakyReLU(LEAKY_ALPHA),
            nn.MaxPool2d(kernel_size=(3,1), stride=(stride,1), padding=(1,0)),
            nn.BatchNorm2d(scale_channels) 
        )
        self.conv1x1 = PointWiseTCN(in_channels, scale_channels, stride=stride)

    def forward(self, x):
        x = torch.cat([self.tcn1(x), self.tcn2(x), self.maxpool3x1(x), self.conv1x1(x)], 1)
        return x
    
    
class Basic_Block(nn.Module):
    def __init__(self, in_channels, out_channels, A, k, eta, kernel_size=5, stride=1, dilations=2, 
                 num_frame=64, num_joint=25, residual=True):
        super(Basic_Block, self).__init__()
        
        num_scale = 4
        scale_channels = out_channels // num_scale
        self.num_scale = num_scale if in_channels !=3 else 1
        
        if in_channels == 3:
            self.gcn = ST_GC(in_channels, out_channels, A)
        else:
            self.gcn = DeSGC(in_channels, 
                             out_channels, 
                             A, 
                             k, 
                             self.num_scale, 
                             num_frame=num_frame, 
                             num_joint=num_joint)
            # self.gcn = CTR_GC(in_channels, 
            #                   out_channels, 
            #                   A, 
            #                   self.num_scale)
        self.tcn = MultiScale_TemporalModeling(out_channels, 
                                               out_channels, 
                                               eta,
                                               stride=stride, 
                                               num_scale=num_scale, 
                                               num_frame=num_frame) 
        
        if in_channels != out_channels:
            self.residual1 = PointWiseTCN(in_channels, out_channels, groups=self.num_scale)
        else:
            self.residual1 = lambda x: x
            
        if not residual:
            self.residual2 = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual2 = lambda x: x
        else:
            self.residual2 = PointWiseTCN(in_channels, out_channels, stride=stride, groups=self.num_scale)
        
        self.relu = nn.LeakyReLU(LEAKY_ALPHA)
        init_param(self.modules())
        
    def forward(self, x):
        res = x
        x = self.gcn(x)
        x = self.relu(x + self.residual1(res))
        x = self.tcn(x)
        x = self.relu(x + self.residual2(res))
        return x
    