import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from argparse import Namespace
from models.multiaba.utils import recover_tokens, combine_tokens, get_distribution_target
from timm.layers import Mlp, DropPath
from timm.models.helpers import adapt_input_conv
from functools import partial
from models.layers.patch_embed import PatchEmbed
from collections import OrderedDict

IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5)
IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)
def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_INCEPTION_MEAN, 'std': IMAGENET_INCEPTION_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


# 定义一个Vision Transformer，采用AbaViT的结构
class AbaViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='', args=None):

        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim # 嵌入维度/特征数量
        self.num_tokens = 2 if distilled else 1

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        # 通过embedding过程，将图像分块，每块投影成向量
        # patch的数量决定了序列的长度
        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.template_sz = 192
        self.search_sz = 384
        template_token_len = int((self.template_sz / patch_size) ** 2)
        search_token_len = int((self.search_sz / patch_size) ** 2)
        self.pos_embed_z = nn.Parameter(torch.zeros(1, template_token_len, 192))
        self.pos_embed_x = nn.Parameter(torch.zeros(1, search_token_len, 192))

        self.reached_token_buffer = None
        self.not_reached_token_buffer = None
        self.combined_mask_buffer = None


        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        # 经过若干个transformer块。这里的transformer更改成了Block_ACT
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer, args=args, index=i, num_patches=self.patch_embed.num_patches+1)
            for i in range(depth)])

        self.norm = norm_layer(embed_dim)

        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        self.eps = 0.01
        for block in self.blocks:
            if args.act_mode == 1:
                torch.nn.init.constant_(block.act_mlp.fc2.bias.data, -1. * args.gate_center)

        self.args = args

        self.rho = None
        self.counter = None
        self.batch_cnt = 0

        self.c_token = None
        self.R_token = None
        self.mask_token = None
        self.rho_token = None
        self.rho_token_weight = None
        self.counter_token = None
        # 总共的token数量， 也就是template和search区域的token数量之和，以及class和dist_token
        self.total_token_cnt = int(3 * template_token_len + search_token_len) + self.num_tokens

        if args.distr_prior_alpha >0. :
            self.distr_target = torch.Tensor(get_distribution_target(standardized=True, target_depth=5)).cuda()
            self.kl_loss = nn.KLDivLoss(reduction='batchmean').cuda()

        self.cat_mode = 'direct'

    def forward_features_act_token(self, z1, z2, z3, x, t_mask=None, s_mask=None):
        # 输入搜索区域的尺寸
        B, H, W = x.shape[0], x.shape[2], x.shape[3]
        if not t_mask is None:
            # 对模板和搜索区域的掩码，按照patch_size进行下采样，以适配patch_embed
            t_mask = t_mask[:,0:t_mask.shape[1]:self.patch_size,0:t_mask.shape[2]:self.patch_size]
            s_mask = s_mask[:,0:s_mask.shape[1]:self.patch_size,0:s_mask.shape[2]:self.patch_size]
            t_mask1 = 1-t_mask
            s_mask1 = 1-s_mask
            # 掩码的有效区域权重为1，无效区域权重是1.5
            t_mask = 1.5*t_mask1 + 1*t_mask
            s_mask = 1.5*s_mask1 + 1*s_mask
            # 将掩码展平，适配transformer给出的序列形式
            t_mask = t_mask.view(t_mask.shape[0],-1)
            s_mask = s_mask.view(s_mask.shape[0],-1)
        ####初始化缓冲区，避免重复定义变量
        bs = x.shape[0]
        if self.reached_token_buffer is None or self.reached_token_buffer.shape[0] != bs:
            self.reached_token_buffer = torch.zeros(bs, self.total_token_cnt, device=x.device)
            self.not_reached_token_buffer = torch.zeros(bs, self.total_token_cnt, device=x.device)
            self.combined_mask_buffer = torch.zeros(bs, self.total_token_cnt, 1, device=x.device)
        ####
        # 投影和位置编码
        x = self.patch_embed(x)
        z1 = self.patch_embed(z1)
        z2 = self.patch_embed(z2)
        z3 = self.patch_embed(z3)

        x += self.pos_embed_x
        z1 += self.pos_embed_z
        z2 += self.pos_embed_z
        z3 += self.pos_embed_z
        # 将搜索区域和目标区域的特征合并
        z12 = combine_tokens(z1, z2, mode=self.cat_mode)
        z = combine_tokens(z12, z3, mode=self.cat_mode)
        x = combine_tokens(z, x, mode=self.cat_mode)
        # 将搜索区域和目标区域的掩码合并
        if not t_mask is None:
            self.rho_token_weight = combine_tokens(t_mask, s_mask, mode=self.cat_mode)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        # 将cls_token和x合并，涉及到是否有dist_token(蒸馏)
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)

        x = self.pos_drop(x)
        bs = x.size()[0]

        if self.c_token is None or bs != self.c_token.size()[0]:
            self.c_token = Variable(torch.zeros(bs, self.total_token_cnt).cuda())
            self.R_token = Variable(torch.ones(bs, self.total_token_cnt).cuda())
            self.mask_token = Variable(torch.ones(bs, self.total_token_cnt).cuda())
            self.rho_token = Variable(torch.zeros(bs, self.total_token_cnt).cuda())
            self.counter_token = Variable(torch.ones(bs, self.total_token_cnt).cuda())

        c_token = self.c_token.clone()
        R_token = self.R_token.clone()
        mask_token = self.mask_token.clone()
        self.rho_token = self.rho_token.detach() * 0.
        self.counter_token = self.counter_token.detach() * 0 + 1.
        output = None
        # 初始化的out是原始序列，经过transformer块后，out会不断更新
        out = x


        if self.args.distr_prior_alpha>0.:
            self.halting_score_layer = []
        # backbone会经历若干个transformer块，在这个过程中l遍历每个块

        

        for i, l in enumerate(self.blocks):
            t1 = time.time()

            mask_expanded = mask_token.float().view(bs, self.total_token_cnt, 1)

            # out.data = out.data * mask_expanded
            out.mul_(mask_expanded)

            # 给forward_act传入一个序列和掩码，返回的是更新后的out和halting_score
            
            t_start = time.time()
            # block_output, h_lst = l.forward(out, 1.-mask_token.float())
            block_output = l.forward_origin(out)
            t_end = time.time()

            print('Block处理时间', t_end-t_start)

        #     # 忽略cls_token后记录当前块的停止分数均值
        #     if self.args.distr_prior_alpha>0.:
        #         self.halting_score_layer.append(torch.mean(h_lst[1][1:]))
        #     # out又会被作为下一个块的输入
        #     out = block_output.clone()
        #     # h_token是当前块每个token的停止分数
        #     _, h_token = h_lst

        #     # 再次对block_output进行掩码
        #     # block_output = block_output * mask_expanded
        #     block_output.mul_(mask_expanded) 

        #     # 对于最后一个块，所有的token的计算都结束了
        #     if i==len(self.blocks)-1:
        #         h_token = Variable(torch.ones(bs, self.total_token_cnt).cuda())

        #     # c_token在每个块中累计停止分数，rho_token记录token的掩码状态
        #     # c_token = c_token + h_token
        #     c_token.add_(h_token)
        #     # self.rho_token = self.rho_token + mask_token.float()
        #     self.rho_token.add_(mask_token)
            
        #     # 如果token的停止分数大于1-eps，说明token的计算应当中止
        #     # reached token是已经达到停止条件，但是还没有被掩码的token
        #     # delta1更新已经达到停止条件的token的输出特征

        #     # reached_token = c_token > 1 - self.eps
        #     # reached_token = reached_token.float() * mask_token.float()
        #     # not_reached_token = c_token < 1 - self.eps
        #     # not_reached_token = not_reached_token.float()            
        #     # delta1 = block_output * R_token.view(bs, self.total_token_cnt, 1) * reached_token.view(bs, self.total_token_cnt, 1)
            
        #     # not_reached_token是还没有达到停止条件的token
        #     # R_token是还没有达到停止条件的token的权重
        #     # delta2更新还没有达到停止条件的token的输出特征
        #     # delta2 = block_output * h_token.view(bs, self.total_token_cnt, 1) * not_reached_token.view(bs, self.total_token_cnt, 1)
        #     # R_token = R_token - (not_reached_token.float() * h_token)            


        # ######
        #     torch.gt(c_token, 1 - self.eps, out=self.reached_token_buffer)
        #     self.reached_token_buffer.mul_(mask_token)
        #     torch.lt(c_token, 1 - self.eps, out=self.not_reached_token_buffer)
        # ######
        #     self.rho_token.addcmul_(R_token, self.reached_token_buffer)
        #     R_token.addcmul_(self.not_reached_token_buffer, h_token, value=-1.0)


        #     # 复用combined_mask_buffer
        #     self.combined_mask_buffer.zero_()
        #     # 计算第一部分
        #     self.combined_mask_buffer.addcmul_(
        #         R_token.view(bs, self.total_token_cnt, 1),
        #         self.reached_token_buffer.view(bs, self.total_token_cnt, 1)
        #     )
        #     # 计算第二部分
        #     self.combined_mask_buffer.addcmul_(
        #         h_token.view(bs, self.total_token_cnt, 1),
        #         self.not_reached_token_buffer.view(bs, self.total_token_cnt, 1)
        #     )
            

        #     # combined_mask = R_token.view(bs, self.total_token_cnt, 1) * reached_token.view(bs, self.total_token_cnt, 1)+ h_token.view(bs, self.total_token_cnt, 1) * not_reached_token.view(bs, self.total_token_cnt, 1)
        #     # self.counter_token = self.counter_token + not_reached_token
        #     delta = block_output * self.combined_mask_buffer
        #     self.counter_token.add_(self.not_reached_token_buffer)
            
        #     # 记录当前token的掩码情况
        #     # mask_token = c_token < 1 - self.eps

            delta = block_output
            if output is None:
                # output = delta1 + delta2
                output = delta
            else:
                #output = output + (delta1 + delta2)
                output.add_(delta)

            t2 = time.time()
            print("一次Transformer时间", t2 - t1)

        x = self.norm(output)

        # 获取模板和搜索区域的向量长度，恢复token
        lens_z = self.pos_embed_z.shape[1]
        lens_x = self.pos_embed_x.shape[1]
        x = recover_tokens(x, lens_z, lens_x, mode=self.cat_mode)

        aux_dict = {"attn": None}
        return x, aux_dict

    def forward(self, z1, z2, z3, x, t_mask=None, s_mask=None):
        if self.args.act_mode == 4:
            x, aux_dict = self.forward_features_act_token(z1, z2, z3,x, t_mask=t_mask, s_mask=s_mask)
        else:
            print('Not implemented yet, please specify for token act.')
            exit()

        return x, aux_dict    

#注册一个按上述定义的模型
def tempabavit_patch16_384(pretrained):
    kwargs = {'num_classes': 1000, 'drop_rate': 0.0, 'drop_path_rate': 0.1}
    model_kwargs = {'act_mode': 4, 'gate_scale': 10.0, 'gate_center': 30.0,'distr_prior_alpha':0.01}
    model_kwargs = Namespace(**model_kwargs)

    kwargs['args'] = model_kwargs
    model = AbaViT(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(pretrained, map_location="cpu")
        ####2025.4.11 更改了加载逻辑，以适应预训练模型的名称变化
        new_state_dict = {}
        is_change = False
        for k, v in checkpoint["model"].items():
            if k.startswith("module.model."):
                is_change = True
                new_key = k[len("module.model."):]  # 去掉前缀
            else:
                new_key = k
            new_state_dict[new_key] = v
        ####
        ####2025.4.16 通过插值将预训练的位置编码也插进来
        pretrain_pos_embed = new_state_dict["pos_embed"]

        pos_embed_z = interpolate_pos_embed(pretrain_pos_embed, target_size=192, embed_dim=192)
        pos_embed_x = interpolate_pos_embed(pretrain_pos_embed, target_size=384, embed_dim=192)
        ####
        if is_change:
            missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
        else:
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint["model"], strict=False)

        param_name = "pos_embed"

            
        print("missing keys:", missing_keys)
        print("unexpected keys:", unexpected_keys)
        print('Load pretrained model from: ' + pretrained)
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["model"], strict=False)
        model.pos_embed_z.data.copy_(pos_embed_z[:, 1:, :])
        model.pos_embed_x.data.copy_(pos_embed_x[:, 1:, :])
        model.temp_token.data.copy_(new_state_dict["cls_token"])
    return model      


# 这是ViT里面一个标准的块结构
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, args=None, index=-1, num_patches=197):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Masked_Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        # 第一个线性层，将维度提升4倍
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        # 从1，2，3，4选择一个mode
        self.act_mode = args.act_mode
        assert self.act_mode in {1, 2, 3, 4}

        self.index=index
        self.args = args

        if self.act_mode == 4:
            self.sig = torch.sigmoid
        else:
            print('Not supported yet.')
            exit()
    # 这是一个普通的Transformer块的forward
    def forward_origin(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    
    # 这是当前实际会使用的forward，每个token加上了一个中止分数
    def forward(self, x, mask=None):
        '''
            输入：x: (B, N, C)  mask: (B, N)标识了不需要参与计算的token
            输出：x: (B, N, C)，经过了一次transformer，  
            halting_score: (B, N)，计算得到每个token的停止分数
        '''
        # x: (B, N, C)
        bs, token, dim = x.shape
        # 没有掩码的情况下，相当于原始的forward方法
        if mask is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
        # mask: (B, N)标识了不需要参与计算的token，扩展到(B, N, 1)，只将需要计算的token传入
            x = x + self.drop_path(self.attn(self.norm1(x*(1-mask).view(bs, token, 1))*(1-mask).view(bs, token, 1), mask=mask))
            x = x + self.drop_path(self.mlp(self.norm2(x*(1-mask).view(bs, token, 1))*(1-mask).view(bs, token, 1)))

        if self.act_mode==4:
            # 两个参数调整特征值的范围和中心，通过sig将序列的第零维映射到[0,1]，作为halting_score
            gate_scale, gate_center = self.args.gate_scale, self.args.gate_center
            halting_score_token = self.sig(x[:,:,0] * gate_scale - gate_center)
            # halting_score: (B, N),代表了每个token的停止分数
            halting_score = [-1, halting_score_token]
        else:
            print('Not supported yet.')
            exit()

        return x, halting_score


# 定义一个可以添加掩码的注意力
class Masked_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., mask=None, masked_softmax_bias=-1000.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.mask = mask
        self.masked_softmax_bias = masked_softmax_bias

    def forward(self, x, mask=None):
        # B:批次大小，N:序列长度，C:隐藏层维度
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            attn = attn + mask.view(mask.shape[0], 1, 1, mask.shape[1]) * self.masked_softmax_bias

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
def interpolate_pos_embed(pretrained_pos_embed, target_size, embed_dim):
    """
    插值调整位置编码
    :param pretrained_pos_embed: 预训练模型的 pos_embed 参数 (1, num_patches+1, embed_dim)
    :param target_size: 目标图像尺寸 (H, W)
    :param embed_dim: 嵌入维度
    :return: 调整后的位置编码
    """
    # 分离类别 token 和 patch token
    cls_token = pretrained_pos_embed[:, :1, :]  # (1, 1, embed_dim)
    patch_pos_embed = pretrained_pos_embed[:, 1:, :]  # (1, num_patches, embed_dim)

    # 计算预训练模型的 patch grid 尺寸
    num_patches = patch_pos_embed.shape[1]
    grid_size = int(num_patches ** 0.5)  # 假设是正方形
    assert grid_size * grid_size == num_patches, "预训练模型的 patch 数量不是正方形"

    # 调整为 (grid_size, grid_size, embed_dim)
    patch_pos_embed = patch_pos_embed.reshape(1, grid_size, grid_size, embed_dim).permute(0, 3, 1, 2)

    # 目标尺寸的 patch grid
    target_grid_size = (target_size // 16, target_size // 16)  # 假设 patch_size=16

    # 使用双线性插值调整尺寸
    patch_pos_embed = F.interpolate(patch_pos_embed, size=target_grid_size, mode='bilinear', align_corners=False)

    # 调整回 (1, num_patches, embed_dim)
    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, -1, embed_dim)

    # 重新组合类别 token 和 patch token
    new_pos_embed = torch.cat((cls_token, patch_pos_embed), dim=1)
    return new_pos_embed