# pytorch_diffusion + derived encoder decoder
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange

from ldm.util import instantiate_from_config
from ldm.modules.attention import LinearAttention
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data, Batch

def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0,1,0,0))
    return emb


def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv, groups=1):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                        groups=groups)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv, groups=1):
        super().__init__()
        self.with_conv = with_conv
        self.groups=groups
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0,
                                        groups=self.groups)

    def forward(self, x):
        if self.with_conv:
            pad = (0,1,0,1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False, groups=1,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.groups = groups

        self.norm1 = Normalize(in_channels, num_groups=32)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1,
                                     groups=self.groups)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels,
                                             out_channels)
        self.norm2 = Normalize(out_channels, num_groups=32)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1,
                                     groups=self.groups)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1,
                                                     groups=self.groups)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0,
                                                    groups=self.groups)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:,:,None,None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h


class LinAttnBlock(LinearAttention):
    """to match AttnBlock usage"""
    def __init__(self, in_channels):
        super().__init__(dim=in_channels, heads=1, dim_head=in_channels)


class AttnBlock(nn.Module):
    def __init__(self, in_channels, groups=1):
        super().__init__()
        self.in_channels = in_channels
        self.groups = groups

        self.norm = Normalize(in_channels, num_groups=32)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0,
                                 groups=self.groups)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0,
                                 groups=self.groups)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0,
                                 groups=self.groups)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0,
                                        groups=self.groups)


    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = q.reshape(b,c,h*w)
        q = q.permute(0,2,1)   # b,hw,c
        k = k.reshape(b,c,h*w) # b,c,hw
        w_ = torch.bmm(q,k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b,c,h*w)
        w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b,c,h,w)

        h_ = self.proj_out(h_)

        return x+h_


def make_attn(in_channels, groups=1, attn_type="vanilla"):
    assert attn_type in ["vanilla", "linear", "none"], f'attn_type {attn_type} unknown'
    print(f"making attention of type '{attn_type}' with {in_channels} in_channels")
    if attn_type == "vanilla":
        return AttnBlock(in_channels, groups)
    elif attn_type == "none":
        return nn.Identity(in_channels)
    else:
        return LinAttnBlock(in_channels)


class Model(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, use_timestep=True, use_linear_attn=False, attn_type="vanilla"):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = self.ch*4
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        self.use_timestep = use_timestep
        if self.use_timestep:
            # timestep embedding
            self.temb = nn.Module()
            self.temb.dense = nn.ModuleList([
                torch.nn.Linear(self.ch,
                                self.temb_ch),
                torch.nn.Linear(self.temb_ch,
                                self.temb_ch),
            ])

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            skip_in = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                if i_block == self.num_res_blocks:
                    skip_in = ch*in_ch_mult[i_level]
                block.append(ResnetBlock(in_channels=block_in+skip_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x, t=None, context=None):
        #assert x.shape[2] == x.shape[3] == self.resolution
        if context is not None:
            # assume aligned context, cat along channel axis
            x = torch.cat((x, context), dim=1)
        if self.use_timestep:
            # timestep embedding
            assert t is not None
            temb = get_timestep_embedding(t, self.ch)
            temb = self.temb.dense[0](temb)
            temb = nonlinearity(temb)
            temb = self.temb.dense[1](temb)
        else:
            temb = None

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](
                    torch.cat([h, hs.pop()], dim=1), temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h

    def get_last_layer(self):
        return self.conv_out.weight

class LIIF_Cube(nn.Module):
    def __init__(self, imnet_in_dim=None, local_ensemble=True, feat_unfold=True, cell_decode=True, node_map_shape=[32,32,32,10]):
        super().__init__()
        self.local_ensemble = local_ensemble
        self.feat_unfold = feat_unfold
        self.cell_decode = cell_decode
        self.imnet_in_dim = imnet_in_dim
        self.node_map_shape = node_map_shape

        if self.feat_unfold:
            imnet_in_dim *= 9
        imnet_in_dim += 3 # attach coord
        if self.cell_decode:
            imnet_in_dim += 3

        self.imnet = nn.Conv2d(imnet_in_dim, self.imnet_in_dim, 1,1,0)

    def query_rgb(self, coord, cell=None):
        # coord: (B,10,N,3) -> (B, 10*32^3, [h,d,v])
        # feat: (B,V,C,H,D)
        B,V,C,H,D = self.feat.size()
        feat = self.feat

        if self.feat_unfold:
            feat = rearrange(feat, 'b v c h d -> b (v c) h d')
            feat = F.unfold(feat, 3, padding=1).view(B, V, C, -1, H, D) # (b v c 9 h d)
            feat = feat.permute(0,2,3,1,4,5) # (b c 9 v h d)
            feat = feat.reshape(B,-1,V,H,D) # (b (c 9) v h d)

        if self.local_ensemble:
            vv_lst = [0]
            vh_lst = [-1, 1]
            vd_lst = [-1, 1]
            eps_shift = 1e-6
        else:
            vd_lst, vh_lst, vv_lst, eps_shift = [0], [0], [0], 0

        # field radius (global: [-1, 1])
        rd = 2 / feat.shape[-1] / 2
        rh = 2 / feat.shape[-2] / 2
        rv = 2 / feat.shape[-3] / 2

        feat_coord = []
        for v in range(0, feat.shape[-3], 1):
            for h in range(0, feat.shape[-2], 1):
                for d in range(0, feat.shape[-1], 1):
                    center_d = d + (1 // 2)
                    center_h = h + (1 // 2)
                    center_v = v + (1 // 2)
                    feat_coord.append([center_h, center_d, center_v])
        feat_coord = np.array(feat_coord)
        feat_coord = torch.from_numpy(feat_coord).type(torch.float32).cuda().view(feat.shape[-3],feat.shape[-2],feat.shape[-1],3) # (v h d 3)
        feat_coord = feat_coord.permute(3,0,1,2).repeat(feat.shape[0],1,1,1,1) # (b 3 v h d)
        feat_coord[:,0] = 2*(feat_coord[:,0] / (feat.shape[-2]-1)) -1 # h
        feat_coord[:,1] = 2*(feat_coord[:,1] / (feat.shape[-1]-1)) -1 # d
        feat_coord[:,2] = 2*(feat_coord[:,2] / (feat.shape[-3]-1)) -1 # v
        feat_coord.clamp_(-1 + 1e-6, 1 - 1e-6) # (b 3 v h d)

        preds = []
        areas = []
        for vd in vd_lst:
            for vh in vh_lst:
                for vv in vv_lst:
                    coord_ = coord.clone() # (B,N,[h,d,v])
                    coord_[:, :, :, -3] += vh * rh + eps_shift
                    coord_[:, :, :, -2] += vd * rd + eps_shift
                    coord_[:, :, :, -1] += vv * rv + 0
                    coord_.clamp_(-1 + 1e-6, 1 - 1e-6) # (B,V,32^3,3)
                    coord_ = coord_.reshape(B,-1,3) # (B,V*32^3,3)
                    q_feat = F.grid_sample(
                        feat, coord_[:,None,None,],
                        mode='nearest', align_corners=False)[:, :, 0, 0, :] \
                        .permute(0, 2, 1) # (B,V*N,C*9)
                    q_coord = F.grid_sample(
                        feat_coord, coord_[:,None,None,],
                        mode='nearest', align_corners=False)[:, :, 0, 0, :] \
                        .permute(0, 2, 1) # (B,V*N,3)
                    q_feat = q_feat.reshape(B,V,self.node_map_shape[0]**3,-1) # (B,V,32*3,C*9)
                    q_coord = q_coord.reshape(B,V,self.node_map_shape[0]**3,-1) # (B,V,32*3,C*9)
                    rel_coord = coord - q_coord # (B,V,N,[h,d,v])
                    rel_coord[:, :, :, 0] *= feat.shape[-2] # h
                    rel_coord[:, :, :, 1] *= feat.shape[-1] # d
                    rel_coord[:, :, :, 2] *= feat.shape[-3] # v
                    inp = torch.cat([q_feat, rel_coord], dim=-1) # (B V (32 32 32) C*9+3)

                    if self.cell_decode:
                        rel_cell = cell.clone()
                        rel_cell[:, :, :, 0] *= feat.shape[-2] # h
                        rel_cell[:, :, :, 1] *= feat.shape[-1] # d
                        rel_cell[:, :, :, 2] *= feat.shape[-3] # v
                        inp = torch.cat([inp, rel_cell], dim=-1) # (B V (32 32 32) C*9+6)

                    inp = inp.permute(0,3,1,2) # (B C*9+6 V (32 32 32))

                    pred = self.imnet(inp) # (B,C,V,N)
                    pred = pred.permute(0,2,3,1) # (B,V,N,C)==(B, V, (32 32 32), C)
                    preds.append(pred)

                    area = torch.abs(rel_coord[:, :, :, 0] * rel_coord[:, :, :, 1] * 1)
                    areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)
        if self.local_ensemble: # dim V does not need ensemble
            t = areas[0]; areas[0] = areas[3]; areas[3] = t
            t = areas[1]; areas[1] = areas[2]; areas[2] = t
        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1) # (B,N,C)
        return ret

    def forward(self, inp, coord, cell):
        self.feat = inp
        return self.query_rgb(coord, cell)

class LIIF_GAT(nn.Module):
    def __init__(self, in_channels=256, out_channels=1, num_heads=1, group_num=10, grid_size=32):
        super(LIIF_GAT, self).__init__()
        self.liif_cube = LIIF_Cube(imnet_in_dim=in_channels, node_map_shape=[grid_size,grid_size,grid_size,group_num])
        self.conv = GATv2Conv(in_channels, out_channels, heads=num_heads)
        self.group_num = group_num
        self.grid_size = grid_size

    def forward(self, x, edge_index, nodes_coord, test=False, return_attention_weights=False):
        '''
        x: feature map (B,10C,H,D)
        nodes_coord: (B,V,N,3) -> (B,10,32^3,[h,d,v])
        edge_index: [[edge_1],[edge_2],...,[edge_N]]
        '''
        B,C,H,D = x.size()
        x = x.view(B,self.group_num,int(C/self.group_num),H,D) # (B,V,C,H,D)
        # x = x.permute(0,2,1,3,4) # (B,C,V,H,D)
        B,V,C,H,D = x.size()

        cell = torch.ones_like(nodes_coord).cuda()
        cell[:, :, :, 0] *= 2 / x.shape[-2] # h
        cell[:, :, :, 1] *= 2 / x.shape[-1] # d
        cell[:, :, :, 2] *= 1 / x.shape[-3] # v 
        
        x = self.liif_cube(x, nodes_coord, cell) # (B,V,N,C)==(B,10,32*32*32,C)

        x = x.reshape(B*V,self.grid_size**3,-1) # (B*V,N,C)
        if not test:
            edge_index = [torch.from_numpy(np.array(item)).permute(1,0).cuda() for sublist in edge_index for item in sublist] # [B*V lists]
        else:
            edge_index = [item.cuda() for sublist in edge_index for item in sublist]
        graph_data_list = [Data(x=x[i,], edge_index=edge_index[i]) for i in range(B*V)]
        batch_graph_data = Batch.from_data_list(graph_data_list)
        x_list = self.conv(batch_graph_data.x, edge_index=batch_graph_data.edge_index) # (B*V*N,C)
        x_list = x_list.reshape(B,V,self.grid_size**3,-1) # (B,V,N,C)
        return x_list

class GraphEncoder(nn.Module):
    def __init__(self, embed_dim, z_channels, groups, ch, ch_mult=(1,2), num_pool=2, grid_size=32, **ignore_kwargs):
        super().__init__()
        self.grid_size = grid_size
        self.num_pool = num_pool
        self.groupConv = []
        self.liifGAT = []
        self.transConv = []
        for d in range(num_pool):
            if d==0:
                self.groupConv.append(torch.nn.Conv2d(groups,
                                                      ch*ch_mult[d]*groups,
                                                      kernel_size=3,stride=2,padding=1,groups=groups).cuda()) 
            else:
                self.groupConv.append(torch.nn.Conv2d(ch*ch_mult[d-1]*groups,
                                                      ch*ch_mult[d]*groups,
                                                      kernel_size=3,stride=2,padding=1,groups=groups).cuda()) 
            self.liifGAT.append(LIIF_GAT(in_channels=ch*ch_mult[d], 
                                        out_channels=ch*ch_mult[d], 
                                        num_heads=1, group_num=groups, grid_size=grid_size).cuda())

        self.finalConv = torch.nn.Conv2d(z_channels*groups, embed_dim, kernel_size=1,stride=1,padding=0).cuda()

        self.squeeze_w_dim = torch.nn.Conv2d(sum(ch * mul for mul in ch_mult)*grid_size*groups, sum(ch * mul for mul in ch_mult)*groups, kernel_size=1, stride=1, padding=0, groups=groups).cuda()

        self.graph_emb = torch.nn.Conv2d(sum(ch * mul for mul in ch_mult)*groups, z_channels*groups, kernel_size=1, stride=1, padding=0, groups=groups).cuda()

    def retrieval_mip(self, x_3d, nodes_coord, mip_idx):
        B,V,C = x_3d.shape[0], x_3d.shape[1], x_3d.shape[2]
        nodes_coord = nodes_coord.reshape(B,-1,3) # (1,10*32^3,3)
        mip_idx_3d = F.grid_sample(mip_idx[:,None,], nodes_coord[:,None,None,], mode='nearest', align_corners=False) # (1,1,1,1,10*32^3)
        mip_idx_3d = mip_idx_3d.reshape(B, -1, self.grid_size, self.grid_size, self.grid_size) # (1,10,32,32,32)
        # Create grid coordinates
        grid_h, grid_d = torch.meshgrid(torch.linspace(-1, 1, self.grid_size), torch.linspace(-1, 1, self.grid_size))  # Grid coordinates
        grid_h = grid_h.unsqueeze(-1).expand(-1, -1, self.grid_size)  # Expand to match tensor size (32,32,32)
        grid_d = grid_d.unsqueeze(-1).expand(-1, -1, self.grid_size)  # Expand to match tensor size (32,32,32)
        grid_h = grid_h.unsqueeze(0).unsqueeze(0).repeat(B, V, 1,1,1).cuda() # (1,10,32,32,32)
        grid_d = grid_d.unsqueeze(0).unsqueeze(0).repeat(B, V, 1,1,1).cuda() # (1,10,32,32,32)
        # Combine grid coordinates with index matrix
        grid = torch.stack((grid_d, grid_h, mip_idx_3d), dim=-1)  # (B, V, D, H, W, 3)==(1,10,32,32,32,3)
        grid = grid.unsqueeze(2).repeat(1,1,C,1,1,1,1) # (1,10,96,32,32,32,3)
        grid = grid.reshape(-1,self.grid_size,self.grid_size,self.grid_size,3) # (1*10*96,32,32,32,3)
        x_3d = x_3d.reshape(-1,1,self.grid_size,self.grid_size,self.grid_size)
        re_x_mip_3d = F.grid_sample(x_3d, grid, mode='nearest', align_corners=False) # (1*10*96,1,32,32,32)
        re_x_mip = torch.max(re_x_mip_3d, dim=-1)[0]
        re_x_mip = re_x_mip.reshape(B,V,C,self.grid_size,self.grid_size) # (1,10,96,32,32)
        return re_x_mip, re_x_mip_3d

    def forward(self, x, edge_index, nodes_coord, inverse_rot=None, mip_idx=None, test=False):
        'x:(B,V,H,D)==(B,10,256,256)'
        'mip_idx:(B,V,H,D)'
        B,V,H,D = x.size()
        g = []

        for d in range(len(self.groupConv)):
            x = self.groupConv[d](x) # (B,10C,h,d)
            g.append(self.liifGAT[d](x,edge_index, nodes_coord, test=test)) # [(B,10,32^3,C),...]

        g = torch.cat(g, dim=-1) # (B,10,32^3,C_sum)
        C_sum = g.shape[-1]
        g = g.permute(0,1,3,2) # (B,10,C_sum,32^3)
        if mip_idx is not None: # cond stage
            g_emb, g_retrieval_mip = self.retrieval_mip(g, nodes_coord, mip_idx) # (1,10,96,32,32)
            g_emb = g_emb.reshape(B,V*C_sum,self.grid_size,self.grid_size)
            g_retrieval_mip = g_retrieval_mip.reshape(B,V,C_sum,self.grid_size,self.grid_size,self.grid_size) # (1,10,96,32,32,32)
        else: # 1st stage
            g = g.reshape(B,V,C_sum,self.grid_size,self.grid_size,self.grid_size) # (1,10,96,32,32,32)
            g_emb = torch.sum(g, dim=-1) # (1,10,96,32,32)
            g_emb = g_emb.reshape(B,V*C_sum,self.grid_size,self.grid_size)

        g_emb = self.graph_emb(g_emb) # (B,V*Z,32,32)
        g_emb_fullViews = g_emb.reshape(B,V,-1,self.grid_size,self.grid_size).sum(2).detach() # (B,V,32,32)

        '# Resample g_emb'
        if g_emb.shape[-1] != x.shape[-1]:
            h_coord = torch.linspace(start=-1,end=1,steps=int(H/2**self.num_pool))  # steps equals to the size of features at the last pooling.
            d_coord = torch.linspace(start=-1,end=1,steps=int(D/2**self.num_pool))  
            coord = torch.stack(torch.meshgrid(h_coord,d_coord,indexing='xy'), dim=-1).view(-1,2).cuda()
            coord = coord.unsqueeze(0).repeat(B,1,1) # (B,h*d,2)
            coord.clamp_(-1 + 1e-6, 1 - 1e-6)
            coord = coord.view(B,int(H/2**self.num_pool),int(H/2**self.num_pool),2) # (B,h,d,2)
            g_emb = F.grid_sample(g_emb, coord, mode='nearest', align_corners=False) # (B,V*Z,h,d)

        'Add g_emb to spatial feature x'
        latent_code_fullViews = x + g_emb # (B,V*Z,h,d)
        # latent_code_separateViews = x
        # latent_code_fullViews = self.finalConv(latent_code_separateViews)
        # latent_code_separateViews = latent_code_separateViews.reshape(B,V,-1,x.shape[-2],x.shape[-1]).sum(2).detach()
        latent_code_fullViews = self.finalConv(latent_code_fullViews)

        if mip_idx is not None: # cond stage
            # return latent_code_fullViews, latent_code_separateViews
            return latent_code_fullViews, g_emb_fullViews, g_retrieval_mip
        else: # 1st stage
            # return latent_code_fullViews, latent_code_separateViews
            return latent_code_fullViews, g_emb_fullViews, g


class Encoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks, groups=1,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, double_z=True, use_linear_attn=False, attn_type="vanilla",
                 **ignore_kwargs):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.groups = groups
        self.resolution = resolution
        self.in_channels = in_channels

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1,
                                       groups=self.groups)

        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout,
                                         groups=self.groups))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, groups=self.groups, attn_type=attn_type))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv, groups=self.groups)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout,
                                       groups=self.groups)
        self.mid.attn_1 = make_attn(block_in, groups=self.groups, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout,
                                       groups=self.groups)

        # end
        self.norm_out = Normalize(block_in, num_groups=32)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        2*z_channels if double_z else z_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                        groups=self.groups)

    def forward(self, x):
        # timestep embedding
        temb = None

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h

class Decoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks, groups=1,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, give_pre_end=False, tanh_out=False, use_linear_attn=False,
                 attn_type="vanilla", **ignorekwargs):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.groups = groups
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,)+tuple(ch_mult)
        block_in = ch*ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)
        self.z_shape = (1,z_channels,curr_res,curr_res)
        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1,
                                       groups=self.groups)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout,
                                       groups=self.groups)
        self.mid.attn_1 = make_attn(block_in, groups=self.groups, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout,
                                       groups=self.groups)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout,
                                         groups=self.groups))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, groups=self.groups, attn_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv, groups=self.groups)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in, num_groups=32)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                        groups=self.groups)

    def forward(self, z):
        #assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        if self.tanh_out:
            h = torch.tanh(h)
        return h


class SimpleDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__()
        self.model = nn.ModuleList([nn.Conv2d(in_channels, in_channels, 1),
                                     ResnetBlock(in_channels=in_channels,
                                                 out_channels=2 * in_channels,
                                                 temb_channels=0, dropout=0.0),
                                     ResnetBlock(in_channels=2 * in_channels,
                                                out_channels=4 * in_channels,
                                                temb_channels=0, dropout=0.0),
                                     ResnetBlock(in_channels=4 * in_channels,
                                                out_channels=2 * in_channels,
                                                temb_channels=0, dropout=0.0),
                                     nn.Conv2d(2*in_channels, in_channels, 1),
                                     Upsample(in_channels, with_conv=True)])
        # end
        self.norm_out = Normalize(in_channels)
        self.conv_out = torch.nn.Conv2d(in_channels,
                                        out_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        for i, layer in enumerate(self.model):
            if i in [1,2,3]:
                x = layer(x, None)
            else:
                x = layer(x)

        h = self.norm_out(x)
        h = nonlinearity(h)
        x = self.conv_out(h)
        return x


class UpsampleDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, ch, num_res_blocks, resolution,
                 ch_mult=(2,2), dropout=0.0):
        super().__init__()
        # upsampling
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        block_in = in_channels
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.res_blocks = nn.ModuleList()
        self.upsample_blocks = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            res_block = []
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                res_block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
            self.res_blocks.append(nn.ModuleList(res_block))
            if i_level != self.num_resolutions - 1:
                self.upsample_blocks.append(Upsample(block_in, True))
                curr_res = curr_res * 2

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        # upsampling
        h = x
        for k, i_level in enumerate(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.res_blocks[i_level][i_block](h, None)
            if i_level != self.num_resolutions - 1:
                h = self.upsample_blocks[k](h)
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class LatentRescaler(nn.Module):
    def __init__(self, factor, in_channels, mid_channels, out_channels, depth=2):
        super().__init__()
        # residual block, interpolate, residual block
        self.factor = factor
        self.conv_in = nn.Conv2d(in_channels,
                                 mid_channels,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1)
        self.res_block1 = nn.ModuleList([ResnetBlock(in_channels=mid_channels,
                                                     out_channels=mid_channels,
                                                     temb_channels=0,
                                                     dropout=0.0) for _ in range(depth)])
        self.attn = AttnBlock(mid_channels)
        self.res_block2 = nn.ModuleList([ResnetBlock(in_channels=mid_channels,
                                                     out_channels=mid_channels,
                                                     temb_channels=0,
                                                     dropout=0.0) for _ in range(depth)])

        self.conv_out = nn.Conv2d(mid_channels,
                                  out_channels,
                                  kernel_size=1,
                                  )

    def forward(self, x):
        x = self.conv_in(x)
        for block in self.res_block1:
            x = block(x, None)
        x = torch.nn.functional.interpolate(x, size=(int(round(x.shape[2]*self.factor)), int(round(x.shape[3]*self.factor))))
        x = self.attn(x)
        for block in self.res_block2:
            x = block(x, None)
        x = self.conv_out(x)
        return x


class MergedRescaleEncoder(nn.Module):
    def __init__(self, in_channels, ch, resolution, out_ch, num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True,
                 ch_mult=(1,2,4,8), rescale_factor=1.0, rescale_module_depth=1):
        super().__init__()
        intermediate_chn = ch * ch_mult[-1]
        self.encoder = Encoder(in_channels=in_channels, num_res_blocks=num_res_blocks, ch=ch, ch_mult=ch_mult,
                               z_channels=intermediate_chn, double_z=False, resolution=resolution,
                               attn_resolutions=attn_resolutions, dropout=dropout, resamp_with_conv=resamp_with_conv,
                               out_ch=None)
        self.rescaler = LatentRescaler(factor=rescale_factor, in_channels=intermediate_chn,
                                       mid_channels=intermediate_chn, out_channels=out_ch, depth=rescale_module_depth)

    def forward(self, x):
        x = self.encoder(x)
        x = self.rescaler(x)
        return x


class MergedRescaleDecoder(nn.Module):
    def __init__(self, z_channels, out_ch, resolution, num_res_blocks, attn_resolutions, ch, ch_mult=(1,2,4,8),
                 dropout=0.0, resamp_with_conv=True, rescale_factor=1.0, rescale_module_depth=1):
        super().__init__()
        tmp_chn = z_channels*ch_mult[-1]
        self.decoder = Decoder(out_ch=out_ch, z_channels=tmp_chn, attn_resolutions=attn_resolutions, dropout=dropout,
                               resamp_with_conv=resamp_with_conv, in_channels=None, num_res_blocks=num_res_blocks,
                               ch_mult=ch_mult, resolution=resolution, ch=ch)
        self.rescaler = LatentRescaler(factor=rescale_factor, in_channels=z_channels, mid_channels=tmp_chn,
                                       out_channels=tmp_chn, depth=rescale_module_depth)

    def forward(self, x):
        x = self.rescaler(x)
        x = self.decoder(x)
        return x


class Upsampler(nn.Module):
    def __init__(self, in_size, out_size, in_channels, out_channels, ch_mult=2):
        super().__init__()
        assert out_size >= in_size
        num_blocks = int(np.log2(out_size//in_size))+1
        factor_up = 1.+ (out_size % in_size)
        print(f"Building {self.__class__.__name__} with in_size: {in_size} --> out_size {out_size} and factor {factor_up}")
        self.rescaler = LatentRescaler(factor=factor_up, in_channels=in_channels, mid_channels=2*in_channels,
                                       out_channels=in_channels)
        self.decoder = Decoder(out_ch=out_channels, resolution=out_size, z_channels=in_channels, num_res_blocks=2,
                               attn_resolutions=[], in_channels=None, ch=in_channels,
                               ch_mult=[ch_mult for _ in range(num_blocks)])

    def forward(self, x):
        x = self.rescaler(x)
        x = self.decoder(x)
        return x


class Resize(nn.Module):
    def __init__(self, in_channels=None, learned=False, mode="bilinear"):
        super().__init__()
        self.with_conv = learned
        self.mode = mode
        if self.with_conv:
            print(f"Note: {self.__class__.__name} uses learned downsampling and will ignore the fixed {mode} mode")
            raise NotImplementedError()
            assert in_channels is not None
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=4,
                                        stride=2,
                                        padding=1)

    def forward(self, x, scale_factor=1.0):
        if scale_factor==1.0:
            return x
        else:
            x = torch.nn.functional.interpolate(x, mode=self.mode, align_corners=False, scale_factor=scale_factor)
        return x

class FirstStagePostProcessor(nn.Module):

    def __init__(self, ch_mult:list, in_channels,
                 pretrained_model:nn.Module=None,
                 reshape=False,
                 n_channels=None,
                 dropout=0.,
                 pretrained_config=None):
        super().__init__()
        if pretrained_config is None:
            assert pretrained_model is not None, 'Either "pretrained_model" or "pretrained_config" must not be None'
            self.pretrained_model = pretrained_model
        else:
            assert pretrained_config is not None, 'Either "pretrained_model" or "pretrained_config" must not be None'
            self.instantiate_pretrained(pretrained_config)

        self.do_reshape = reshape

        if n_channels is None:
            n_channels = self.pretrained_model.encoder.ch

        self.proj_norm = Normalize(in_channels,num_groups=in_channels//2)
        self.proj = nn.Conv2d(in_channels,n_channels,kernel_size=3,
                            stride=1,padding=1)

        blocks = []
        downs = []
        ch_in = n_channels
        for m in ch_mult:
            blocks.append(ResnetBlock(in_channels=ch_in,out_channels=m*n_channels,dropout=dropout))
            ch_in = m * n_channels
            downs.append(Downsample(ch_in, with_conv=False))

        self.model = nn.ModuleList(blocks)
        self.downsampler = nn.ModuleList(downs)


    def instantiate_pretrained(self, config):
        model = instantiate_from_config(config)
        self.pretrained_model = model.eval()
        # self.pretrained_model.train = False
        for param in self.pretrained_model.parameters():
            param.requires_grad = False


    @torch.no_grad()
    def encode_with_pretrained(self,x):
        c = self.pretrained_model.encode(x)
        if isinstance(c, DiagonalGaussianDistribution):
            c = c.mode()
        return  c

    def forward(self,x):
        z_fs = self.encode_with_pretrained(x)
        z = self.proj_norm(z_fs)
        z = self.proj(z)
        z = nonlinearity(z)

        for submodel, downmodel in zip(self.model,self.downsampler):
            z = submodel(z,temb=None)
            z = downmodel(z)

        if self.do_reshape:
            z = rearrange(z,'b c h w -> b (h w) c')
        return z

