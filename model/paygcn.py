import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt

from .modules import Basic_Block
from graph.mhr70_def import pose_info

def import_class(name):
    components = name.split(".")
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


def build_mhr70_edges():
    """
    Returns:
        edges: list[(u,v)] 0-based joint index edges
    """
    kinfo = pose_info["keypoint_info"]
    sinfo = pose_info["skeleton_info"]

    name_to_id = {v["name"]: v["id"] for _, v in kinfo.items()}

    edges = []
    for _, item in sinfo.items():
        a, b = item["link"]
        edges.append((name_to_id[a], name_to_id[b]))
    return edges


# ============================================================
# DeGCN backbone
# ============================================================
class DeGCN(nn.Sequential):
    def __init__(self, block_args, A, k, eta):
        super(DeGCN, self).__init__()
        for i, [in_channels, out_channels, stride, residual, num_frame, num_joint] in enumerate(block_args):
            self.add_module(
                f"block-{i}_tcngcn",
                Basic_Block(
                    in_channels,
                    out_channels,
                    A,
                    k,
                    eta,
                    stride=stride,
                    num_frame=num_frame,
                    num_joint=num_joint,
                    residual=residual,
                ),
            )


# ============================================================
# Cross-attn module
# ============================================================
class CrossAttentionBlock(nn.Module):
    def __init__(self, dim_q, dim_kv, dim_hidden=256, num_heads=4, dropout=0.1):
        super().__init__()
        assert dim_hidden % num_heads == 0
        self.num_heads = num_heads
        self.dim_hidden = dim_hidden
        self.dh = dim_hidden // num_heads

        self.norm_q = nn.LayerNorm(dim_q)
        self.q_proj = nn.Linear(dim_q, dim_hidden)
        self.k_proj = nn.Linear(dim_kv, dim_hidden)
        self.v_proj = nn.Linear(dim_kv, dim_hidden)
        self.out_proj = nn.Linear(dim_hidden, dim_q)
        self.drop = nn.Dropout(dropout)

    def forward(self, q, kv):
        N, Lq, _ = q.shape
        _, Lk, _ = kv.shape
        H, Dh = self.num_heads, self.dh

        q0 = q
        q = self.norm_q(q)

        Q = self.q_proj(q).view(N, Lq, H, Dh).transpose(1, 2)    # (N,H,Lq,Dh)
        K = self.k_proj(kv).view(N, Lk, H, Dh).transpose(1, 2)   # (N,H,Lk,Dh)
        V = self.v_proj(kv).view(N, Lk, H, Dh).transpose(1, 2)   # (N,H,Lk,Dh)

        attn = (Q @ K.transpose(-2, -1)) / (Dh ** 0.5)
        attn = attn.softmax(dim=-1)
        attn = self.drop(attn)

        out = attn @ V
        out = out.transpose(1, 2).contiguous().view(N, Lq, H * Dh)
        out = self.out_proj(out)

        return q0 + out


class DualStreamCrossModalFusion(nn.Module):
    """
    Fs: (N, Cs, Ts, Vs)
    Fr: (N, Cr, Hr, Wr)
    """

    def __init__(self, Cs, Cr, num_class, hidden=256, heads=4):
        super().__init__()
        self.skel_to_rgb = CrossAttentionBlock(dim_q=Cs, dim_kv=Cr, dim_hidden=hidden, num_heads=heads)
        self.rgb_to_skel = CrossAttentionBlock(dim_q=Cr, dim_kv=Cs, dim_hidden=hidden, num_heads=heads)

        # logits from interactive features
        self.fc_skel_inter = nn.Linear(Cs, num_class)
        self.fc_rgb_inter = nn.Linear(Cr, num_class)

        # gate fusion
        self.gate_fc = nn.Linear(Cs + Cr, 2)

    def forward(self, Fs, Fr):
        N, Cs, Ts, Vs = Fs.shape
        _, Cr, Hr, Wr = Fr.shape

        # tokens
        skel_tokens = Fs.mean(dim=2).permute(0, 2, 1).contiguous()           # (N, Vs, Cs)
        rgb_tokens = Fr.view(N, Cr, Hr * Wr).permute(0, 2, 1).contiguous()   # (N, Lr, Cr)

        # cross-attn
        skel_tokens2 = self.skel_to_rgb(skel_tokens, rgb_tokens)
        rgb_tokens2 = self.rgb_to_skel(rgb_tokens, skel_tokens)

        # pooling
        f_skel = skel_tokens2.mean(dim=1)   # (N, Cs)
        f_rgb = rgb_tokens2.mean(dim=1)     # (N, Cr)

        # interactive logits
        logit_skel_inter = self.fc_skel_inter(f_skel)
        logit_rgb_inter = self.fc_rgb_inter(f_rgb)

        # gate fusion
        gate = F.softmax(self.gate_fc(torch.cat([f_skel, f_rgb], dim=-1)), dim=-1)
        logit_fused = gate[:, 0:1] * logit_skel_inter + gate[:, 1:2] * logit_rgb_inter

        return logit_fused, logit_skel_inter, logit_rgb_inter, gate

# ============================================================
# RGB backbone
# ============================================================
class ResNet18Backbone(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        from torchvision.models import resnet18
        m = resnet18(pretrained=pretrained)
        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
        self.layer1 = m.layer1
        self.layer2 = m.layer2
        self.layer3 = m.layer3
        self.layer4 = m.layer4

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x  # (N,512,H/32,W/32)


# ============================================================
# SDD: Spatial Difference Discriminator
#   - learn two anchors by attention pooling over joints
#   - build hand-to-anchor difference vectors and temporal conv -> logit
# ============================================================
class SpatialDifferenceDiscriminator(nn.Module):
    def __init__(
        self,
        Cs: int,
        num_class: int,
        left_hand_idx: int = 30,
        right_hand_idx: int = 50,
        tcn_hidden: int = 256,
        dropout: float = 0.2,
        debug_dir: str = "./sdd_debug",
    ):
        super().__init__()
        self.left_hand_idx = int(left_hand_idx)
        self.right_hand_idx = int(right_hand_idx)
        self.debug_dir = debug_dir
        self.skeleton_edges = build_mhr70_edges()

        def make_attn_head():
            return nn.Sequential(
                nn.Linear(Cs, max(Cs // 4, 32)),
                nn.ReLU(inplace=True),
                nn.Linear(max(Cs // 4, 32), 1),
            )

        self.attn_left = make_attn_head()
        self.attn_right = make_attn_head()

        in_ch = 14
        self.tcn = nn.Sequential(
            nn.Conv1d(in_ch, tcn_hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(tcn_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Conv1d(tcn_hidden, tcn_hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(tcn_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.fc = nn.Linear(tcn_hidden, num_class)

    @staticmethod
    def _safe_norm(x, eps=1e-6):
        return torch.sqrt(torch.clamp((x * x).sum(dim=-1, keepdim=True), min=eps))

    def _attn_anchor(self, X, Fs, attn_head):
        """
        X:  (N, T, V, 3)
        Fs: (N, Cs, Ts, Vs)
        return:
          p: (N, T, 3)
          w: (N, V)
        """
        joint_feat = Fs.mean(dim=2).permute(0, 2, 1).contiguous()  # (N, V, Cs)
        logits = attn_head(joint_feat).squeeze(-1)                 # (N, V)
        w = F.softmax(logits, dim=-1)                              # (N, V)

        p = torch.einsum("ntvc,nv->ntc", X, w)  # (N, T, 3)
        return p, w

    @torch.no_grad()
    def _save_debug_plot(self, X, pL, pR, wL, wR, epoch=0, step=0, sample_idx=0, frame_idx=0):
        """
        X:  (N, T, V, 3)
        pL: (N, T, 3)
        pR: (N, T, 3)
        wL: (N, V)
        wR: (N, V)
        """
        os.makedirs(self.debug_dir, exist_ok=True)

        X_np = X[sample_idx, frame_idx].detach().cpu().numpy()     # (V,3)
        pL_np = pL[sample_idx, frame_idx].detach().cpu().numpy()   # (3,)
        pR_np = pR[sample_idx, frame_idx].detach().cpu().numpy()   # (3,)

        wL_np = wL[sample_idx].detach().cpu().numpy()
        wR_np = wR[sample_idx].detach().cpu().numpy()

        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection="3d")

        # skeleton edges
        for (u, v) in self.skeleton_edges:
            if u < X_np.shape[0] and v < X_np.shape[0]:
                ax.plot(
                    [X_np[u, 0], X_np[v, 0]],
                    [X_np[u, 1], X_np[v, 1]],
                    [X_np[u, 2], X_np[v, 2]],
                    linewidth=1,
                    alpha=0.5,
                )

        # skeleton joints
        ax.scatter(X_np[:, 0], X_np[:, 1], X_np[:, 2], s=12, alpha=0.6, label="Joints")

        # left/right finger joints
        if self.left_hand_idx < X_np.shape[0]:
            xL = X_np[self.left_hand_idx]
            ax.scatter(xL[0], xL[1], xL[2], s=80, marker="o", label=f"Left finger (id={self.left_hand_idx})")

        if self.right_hand_idx < X_np.shape[0]:
            xR = X_np[self.right_hand_idx]
            ax.scatter(xR[0], xR[1], xR[2], s=80, marker="o", label=f"Right finger (id={self.right_hand_idx})")

        # anchors
        ax.scatter(pL_np[0], pL_np[1], pL_np[2], s=150, marker="*", label="Anchor-L")
        ax.scatter(pR_np[0], pR_np[1], pR_np[2], s=150, marker="*", label="Anchor-R")

        ax.set_title(f"SDD Debug (epoch={epoch}, step={step}, frame={frame_idx})")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend(loc="upper right")

        save_path = os.path.join(self.debug_dir, f"sdd_e{epoch}_s{step}_f{frame_idx}.png")
        print(f"Saving SDD debug plot to: {save_path}")
        plt.savefig(save_path, dpi=200)
        plt.close()

    def forward(self, X, Fs, return_debug=False, debug_epoch=0, debug_step=0):
        """
        X:  (N, T, V, 3)
        Fs: (N, Cs, Ts, Vs)
        """
        N, T, V, _ = X.shape

        pL, wL = self._attn_anchor(X, Fs, self.attn_left)
        pR, wR = self._attn_anchor(X, Fs, self.attn_right)
        pL = pL.mean(dim=1)[:, None, :].expand(-1, T, -1)  # (N,T,3) # average over time Ts
        pR = pR.mean(dim=1)[:, None, :].expand(-1, T, -1)  # (N,T,3) # average over time Ts

        if self.left_hand_idx >= V or self.right_hand_idx >= V:
            raise ValueError(
                f"Hand joint idx out of range: left={self.left_hand_idx}, right={self.right_hand_idx}, V={V}"
            )

        xL = X[:, :, self.left_hand_idx, :]   # (N,T,3)
        xR = X[:, :, self.right_hand_idx, :]  # (N,T,3)

        dL = xL - pL
        dR = xR - pR

        ddL = torch.zeros_like(dL)
        ddR = torch.zeros_like(dR)
        ddL[:, 1:, :] = dL[:, 1:, :] - dL[:, :-1, :]
        ddR[:, 1:, :] = dR[:, 1:, :] - dR[:, :-1, :]

        rL = self._safe_norm(dL)  # (N,T,1)
        rR = self._safe_norm(dR)  # (N,T,1)

        feat = torch.cat([dL, ddL, rL, dR, ddR, rR], dim=-1)  # (N,T,14)

        feat = feat.permute(0, 2, 1).contiguous()  # (N,14,T)
        z = self.tcn(feat)                         # (N,H,T)
        z = z.mean(dim=-1)                         # GAP over time
        logit = self.fc(z)                         # (N,num_class)

        # ---- auto save debug plot if requested ----
        if return_debug:
            # plot frame middle
            mid_t = T // 2
            self._save_debug_plot(
                X, pL, pR, wL, wR,
                epoch=debug_epoch,
                step=debug_step,
                sample_idx=0,
                frame_idx=mid_t,
            )
            return logit, pL, pR, wL, wR

        return logit


# ============================================================
# Full Model: Skeleton streams + RGB stream + Fusion stream + SDD
# ============================================================
class Model(nn.Module):
    def __init__(
        self,
        num_class=60,
        num_point=25,
        num_person=2,
        k=8,
        eta=4,
        num_stream=2,
        graph=None,
        graph_args=dict(),
        in_channels=3,
        drop_out=0,

        # rgb settings
        use_rgb=True,
        rgb_pretrained=True,

        # fusion settings
        use_fusion=True,
        fusion_on_stream_idx=-1,

        fusion_hidden=256,
        fusion_heads=4,

        # SDD settings
        use_sdd=True,
        sdd_stream_idx=0,  # which skeleton stream feature map to build SDD anchor
        left_hand_idx=30,
        right_hand_idx=50,
        sdd_tcn_hidden=256,
        sdd_dropout=0.2,
    ):
        super(Model, self).__init__()

        if graph is None:
            raise ValueError("graph must be provided")
        Graph = import_class(graph)
        self.graph = Graph(**graph_args)
        A = self.graph.A

        self.num_class = num_class
        self.num_point = num_point
        self.num_person = num_person

        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        base_channel = 64
        base_frame = 64

        self.blockargs = [
            [in_channels, base_channel, 1, False, base_frame, num_point],
            [base_channel, base_channel, 1, True, base_frame, num_point],
            [base_channel, base_channel, 1, True, base_frame, num_point],
            [base_channel, base_channel, 1, True, base_frame, num_point],
            [base_channel, base_channel * 2, 2, True, base_frame, num_point],
            [base_channel * 2, base_channel * 2, 1, True, base_frame // 2, num_point],
            [base_channel * 2, base_channel * 2, 1, True, base_frame // 2, num_point],
            [base_channel * 2, base_channel * 4, 2, True, base_frame // 2, num_point],
            [base_channel * 4, base_channel * 4, 1, True, base_frame // 4, num_point],
            [base_channel * 4, base_channel * 4, 1, True, base_frame // 4, num_point],
        ]

        # skeleton streams
        self.num_stream = num_stream
        self.streams = nn.ModuleList([DeGCN(self.blockargs, A, k, eta) for _ in range(num_stream)])
        self.fc_skel = nn.ModuleList([nn.Linear(base_channel * 4, num_class) for _ in range(num_stream)])

        for fc in self.fc_skel:
            nn.init.normal_(fc.weight, 0, math.sqrt(2.0 / num_class))

        bn_init(self.data_bn, 1)

        self.drop_out = nn.Dropout(drop_out) if drop_out else (lambda x: x)

        # rgb stream
        self.use_rgb = use_rgb
        if self.use_rgb:
            self.rgb_backbone = ResNet18Backbone(pretrained=rgb_pretrained)
            self.rgb_fc = nn.Linear(512, num_class)

        # fusion stream
        self.use_fusion = use_fusion
        self.fusion_on_stream_idx = fusion_on_stream_idx
        if self.use_fusion:
            self.fusion = DualStreamCrossModalFusion(
                Cs=base_channel * 4,  # 256
                Cr=512,
                num_class=num_class,
                hidden=fusion_hidden,
                heads=fusion_heads,
            )

        # SDD
        self.use_sdd = use_sdd
        self.sdd_stream_idx = sdd_stream_idx
        if self.use_sdd:
            self.sdd = SpatialDifferenceDiscriminator(
                Cs=base_channel * 4,
                num_class=num_class,
                left_hand_idx=left_hand_idx,
                right_hand_idx=right_hand_idx,
                tcn_hidden=sdd_tcn_hidden,
                dropout=sdd_dropout,
            )

    def forward(self, x_skel, x_rgb=None):
        # ==================================================
        # Skeleton forward (same preprocessing as yours)
        # ==================================================
        if len(x_skel.shape) == 3:
            N, T, VC = x_skel.shape
            x_skel = x_skel.view(N, T, self.num_point, -1).permute(0, 3, 1, 2).contiguous().unsqueeze(-1)

        # x_skel: (N,C,T,V,M)
        N, C, T, V, M = x_skel.size()

        # ---- keep a copy of joint coords for SDD ----
        # Use first 3 channels as xyz if available; otherwise use all channels (but SDD assumes 3D)
        if C < 3:
            raise ValueError(f"SDD requires at least 3 channels for joint coords, but got C={C}")
        X = x_skel[:, 0:3, :, :, :]               # (N,3,T,V,M)
        X = X.mean(dim=-1)                        # merge persons -> (N,3,T,V)
        X = X.permute(0, 2, 3, 1).contiguous()     # (N,T,V,3)

        x = x_skel.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        x_ = x

        out_skel_list = []
        featmap_list = []

        for stream, fc in zip(self.streams, self.fc_skel):
            xs = x_
            xs = stream(xs)  # (N*M, 256, Ts, Vs)
            featmap_list.append(xs)

            c_new = xs.size(1)
            xs_pool = xs.view(N, M, c_new, -1).mean(3).mean(1)  # (N,256)
            xs_pool = self.drop_out(xs_pool)
            out_skel_list.append(fc(xs_pool))

        # ==================================================
        # RGB stream forward
        # ==================================================
        logit_rgb = None
        Fr = None
        if self.use_rgb and (x_rgb is not None):
            Fr = self.rgb_backbone(x_rgb)  # (N,512,H,W)
            f_rgb = Fr.mean(dim=[2, 3])    # GAP -> (N,512)
            f_rgb = self.drop_out(f_rgb)
            logit_rgb = self.rgb_fc(f_rgb)

        # ==================================================
        # Fusion stream forward
        # ==================================================
        logit_fused = None
        if self.use_fusion and (Fr is not None):
            Fs_nm = featmap_list[self.fusion_on_stream_idx]  # (N*M,256,Ts,Vs)
            _, Cs, Ts, Vs = Fs_nm.shape
            Fs = Fs_nm.view(N, M, Cs, Ts, Vs).mean(dim=1)     # (N,256,Ts,Vs)
            logit_fused, _, _, _ = self.fusion(Fs, Fr)            

        # ==================================================
        # SDD branch forward
        # ==================================================
        logit_sdd = None
        if self.use_sdd:
            Fs_nm = featmap_list[self.sdd_stream_idx]          # (N*M,256,Ts,Vs)
            _, Cs, Ts, Vs = Fs_nm.shape
            Fs = Fs_nm.view(N, M, Cs, Ts, Vs).mean(dim=1)      # (N,256,Ts,Vs)
            # debug only in eval
            if (not self.training):
                logit_sdd, pL, pR, wL, wR = self.sdd(
                    X, Fs,
                    return_debug=True,
                    debug_epoch=0,
                    debug_step=0
                )
            else:
                logit_sdd = self.sdd(X, Fs)                    # (N,num_class)

        # return all logits (filter None)
        result = out_skel_list + [logit_rgb, logit_fused, logit_sdd]
        return [x for x in result if x is not None]
