"""
Extended from ADNet code by Hansen et al.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from models.encoder import Res101Encoder
import numpy as np
from models.Modules import Decoder, InfoAggregation
import random


class FewShotSeg(nn.Module):

    def __init__(self, pretrained_weights="deeplabv3"):
        super().__init__()

        # Encoder
        self.encoder = Res101Encoder(replace_stride_with_dilation=[True, True, False],
                                     pretrained_weights=pretrained_weights)  # or "resnet101"
        self.device = torch.device('cuda')
        self.scaler = 10.0
        self.my_weight = torch.FloatTensor([0.1, 1.0]).cuda()
        self.criterion = nn.NLLLoss(ignore_index=255, weight=self.my_weight)
        self.rate = 0.95
        self.kernel = (16, 16)
        self.stride = 8
        self.y = 3.5
        self.a = Parameter(torch.Tensor([0.5]))
        self.beta1 = Parameter(torch.Tensor([0.5]))
        self.beta2 = Parameter(torch.Tensor([0.5]))
        self.fc = nn.Linear(1, 1, bias=True)
        self.decoder1 = Decoder()
        self.decoder2 = Decoder()
        self.IA1 = InfoAggregation(512, 512)
        self.IA2 = InfoAggregation(512, 512)

    def forward(self, supp_imgs, supp_mask, qry_imgs, train=False):
        """
        Args:
            supp_imgs: support images
                way x shot x [B x 3 x H x W], list of lists of tensors
            fore_mask: foreground masks for support images
                way x shot x [B x H x W], list of lists of tensors
            back_mask: background masks for support images
                way x shot x [B x H x W], list of lists of tensors
            qry_imgs: query images
                N x [B x 3 x H x W], list of tensors
        """

        self.n_ways = len(supp_imgs)
        self.n_shots = len(supp_imgs[0])
        self.n_queries = len(qry_imgs)
        assert self.n_ways == 1  # for now only one-way, because not every shot has multiple sub-images
        assert self.n_queries == 1

        qry_bs = qry_imgs[0].shape[0]
        supp_bs = supp_imgs[0][0].shape[0]
        img_size = supp_imgs[0][0].shape[-2:]
        supp_mask = torch.stack([torch.stack(way, dim=0) for way in supp_mask],
                                dim=0).view(supp_bs, self.n_ways, self.n_shots, *img_size)  # B x Wa x Sh x H x W

        ###### Extract features ######
        imgs_concat = torch.cat([torch.cat(way, dim=0) for way in supp_imgs]
                                + [torch.cat(qry_imgs, dim=0), ], dim=0)

        # encoder output
        img_fts, tao = self.encoder(imgs_concat)
        supp_fts = [img_fts[dic][:self.n_ways * self.n_shots * supp_bs].view(  # B x Wa x Sh x C x H' x W'
            supp_bs, self.n_ways, self.n_shots, -1, *img_fts[dic].shape[-2:]) for _, dic in enumerate(img_fts)]

        qry_fts = [img_fts[dic][self.n_ways * self.n_shots * supp_bs:].view(  # B x N x C x H' x W'
            qry_bs, self.n_queries, -1, *img_fts[dic].shape[-2:]) for _, dic in enumerate(img_fts)]

        ##### Get threshold #######
        self.thresh_pred1 = tao[1:]  # query
        self.thresh_pred2 = self.fc(self.thresh_pred1[:, [1]])
        self.t = tao[0]  # support

        ###### Compute loss ######
        align_loss = torch.zeros(1).to(self.device)
        outputs = []
        for epi in range(supp_bs):
            supp_fts_ = [[F.interpolate(supp_fts[0][[epi], way, shot], size=img_size, mode='bilinear', align_corners=True)
                          for shot in range(self.n_shots)] for way in range(self.n_ways)]
            # get global and local prototypes
            if supp_mask[epi][0].sum() == 0:
                region_protos = [[self.scan(supp_fts_[way][shot], supp_mask[[epi], way, shot], self.kernel, self.stride, flag=0)
                                  for shot in range(self.n_shots)] for way in range(self.n_ways)]  # (way, shot, 2, (n_pts, 512))
            else:
                region_protos = [[self.scan(supp_fts_[way][shot], supp_mask[[epi], way, shot], self.kernel, self.stride, flag=1)
                                  for shot in range(self.n_shots)] for way in range(self.n_ways)]

            bg_pts = [torch.cat([region_protos[way][0][0]], dim=0) for way in range(self.n_ways)]
            fg_pts = [torch.cat([region_protos[way][0][1]], dim=0) for way in range(self.n_ways)]
            # get prior mask
            qry_bg_masks = [self.get_qry_masks(bg_pts[way], qry_fts[0][epi], self.t[[0]]) for way in range(self.n_ways)]
            qry_fg_masks = [self.get_qry_masks(fg_pts[way], qry_fts[0][epi], self.t[[1]]) for way in range(self.n_ways)]
            # refine high-confidence prior mask
            qry_bg_masks_ = [qry_bg_masks[way] * (1-qry_fg_masks[way]) for way in range(self.n_ways)]
            qry_fg_masks_ = [qry_fg_masks[way] * (1-qry_bg_masks[way]) for way in range(self.n_ways)]
            # mine target regions
            qry_fg_fts = self.get_qry_fts(qry_fts[0][epi], supp_fts[0][[epi], 0, 0], supp_mask[[epi], 0, 0], qry_fg_masks_[0])
            qry_bg_fts = self.get_qry_bg_fts(qry_fts[0][epi], supp_fts[0][[epi], 0, 0], supp_mask[[epi], 0, 0], qry_bg_masks_[0])

            # POEM
            matched_fts = torch.stack(
                [self.match_pts(fg_pts[way], qry_fg_fts, qry_fg_masks_[way], self.thresh_pred1[:, [1]]) for way in range(self.n_ways)], dim=1)
            # get fg_preds
            fg_preds = torch.cat([self.decoder1(matched_fts[:, way]) for way in range(self.n_ways)], dim=1)   # [9, way, 64, 64] [1, 1, 64, 64]

            # POEM
            bg_matched_fts = torch.stack(
                [self.match_pts(bg_pts[way], qry_bg_fts, qry_bg_masks_[way], self.thresh_pred2) for way in range(self.n_ways)], dim=1)
            # get bg_preds
            bg_preds = torch.cat([self.decoder2(bg_matched_fts[:, way]) for way in range(self.n_ways)], dim=1)

            fg_preds = F.interpolate(fg_preds, size=img_size, mode='bilinear', align_corners=True)
            bg_preds = F.interpolate(bg_preds, size=img_size, mode='bilinear', align_corners=True)

            preds = torch.cat([bg_preds, fg_preds], dim=1)
            preds = torch.softmax(preds, dim=1)

            outputs.append(preds)
            ''' Prototype alignment loss '''
            if train:
                align_loss_epi = self.alignLoss([supp_fts[n][epi] for n in range(len(supp_fts))],
                                                 [qry_fts[n][epi] for n in range(len(qry_fts))], preds, supp_mask[epi])
                align_loss += align_loss_epi

        output = torch.stack(outputs, dim=1)  # N x B x (1 + Wa) x H x W
        output = output.view(-1, *output.shape[2:])

        return output, align_loss / supp_bs


    def get_masked_fts(self, fts, mask):

        masked_fts = torch.sum(fts * mask[None, ...], dim=(-2, -1)) \
                     / (mask[None, ...].sum(dim=(-2, -1)) + 1e-5)  # 1 x C

        return masked_fts

    def alignLoss(self, supp_fts, qry_fts, pred, fore_mask):
        """
            supp_fts: (2, [1, 512, 64, 64])
            qry_fts: (2, (1, 512, 64, 64))
            pred: [1, 2, 256, 256]
            fore_mask: [Way, Shot , 256, 256]
        """
        n_ways, n_shots = len(fore_mask), len(fore_mask[0])

        # Get query mask
        pred_mask = pred.argmax(dim=1, keepdim=True).squeeze(1)  # N x H' x W'
        binary_masks = [pred_mask == i for i in range(1 + n_ways)]
        skip_ways = [i for i in range(n_ways) if binary_masks[i + 1].sum() == 0]
        pred_mask = torch.stack(binary_masks, dim=0).float()  # (1 + Wa) x N x H' x W'

        # Compute the support loss
        loss = torch.zeros(1).to(self.device)
        for way in range(n_ways):
            if way in skip_ways:
                continue
            # Get the query prototypes
            for shot in range(n_shots):

                qry_fts_ = F.interpolate(qry_fts[0], size=fore_mask.shape[-2:], mode='bilinear', align_corners=True)

                # get region_qry_protos
                if pred_mask[way + 1].sum() == 0:
                    region_qry_protos = [self.scan(qry_fts_, pred_mask[way + 1], self.kernel, self.stride, flag=0)]
                else:
                    region_qry_protos = [self.scan(qry_fts_, pred_mask[way + 1], self.kernel, self.stride, flag=1)]

                qry_bg_pts = [torch.cat([region_qry_protos[0][0]], dim=0)]
                qry_fg_pts = [torch.cat([region_qry_protos[0][1]], dim=0)]

                sup_bg_masks = [self.get_qry_masks(qry_bg_pts[way], supp_fts[0][way, [shot]], self.t[[0]])]
                sup_fg_masks = [self.get_qry_masks(qry_fg_pts[way], supp_fts[0][way, [shot]], self.t[[1]])]
                sup_bg_masks_ = [sup_bg_masks[way] * (1 - sup_fg_masks[way]) for way in range(self.n_ways)]
                sup_fg_masks_ = [sup_fg_masks[way] * (1 - sup_bg_masks[way]) for way in range(self.n_ways)]
                sup_fg_fts = self.get_qry_fts(supp_fts[0][way, [shot]], qry_fts[0], pred_mask[way + 1], sup_fg_masks_[0])
                sup_bg_fts = self.get_qry_bg_fts(supp_fts[0][way, [shot]], qry_fts[0], pred_mask[way + 1], sup_bg_masks_[0])

                # POEM
                matched_fts = torch.stack([self.match_pts(qry_fg_pts[0], sup_fg_fts, sup_fg_masks_[0], self.thresh_pred1[:, [1]])], dim=1)
                # get fg_preds
                fg_preds = torch.cat([self.decoder1(matched_fts[:, 0])], dim=1)

                # POEM
                bg_matched_fts = torch.stack([self.match_pts(qry_bg_pts[0], sup_bg_fts, sup_bg_masks_[0], self.thresh_pred2)], dim=1)
                ### get bg_preds
                bg_preds = torch.cat([self.decoder2(bg_matched_fts[:, 0])], dim=1)

                fg_preds = F.interpolate(fg_preds, size=fore_mask.shape[-2:], mode='bilinear', align_corners=True)
                bg_preds = F.interpolate(bg_preds, size=fore_mask.shape[-2:], mode='bilinear', align_corners=True)

                pred_ups = torch.cat([bg_preds, fg_preds], dim=1)
                pred_ups = torch.softmax(pred_ups, dim=1)

                # Construct the support Ground-Truth segmentation
                supp_label = torch.full_like(fore_mask[way, shot], 255, device=fore_mask.device)
                supp_label[fore_mask[way, shot] == 1] = 1
                supp_label[fore_mask[way, shot] == 0] = 0

                # Compute Loss
                eps = torch.finfo(torch.float32).eps
                log_prob = torch.log(torch.clamp(pred_ups, eps, 1 - eps))
                loss += self.criterion(log_prob, supp_label[None, ...].long()) / n_shots / n_ways

        return loss

    def get_pts(self, fts, mask):
        features_trans = fts.squeeze(0)
        features_trans = features_trans.permute(1, 2, 0)
        features_trans = features_trans.view(features_trans.shape[-2] * features_trans.shape[-3],
                                             features_trans.shape[-1])
        mask = mask.squeeze(0).view(-1)
        indx = mask == 1
        features_trans = features_trans[indx]
        if len(features_trans) > 100:
            features_trans = features_trans[random.sample(range(len(features_trans)), 100)]
        return features_trans

    def scan(self, fts, mask, kernel_size=(16, 16), stride=8, flag=1):
        # get size of sliding window
        b, c, height, width = fts.size()
        kernel_h, kernel_w = kernel_size
        n_fg = mask.sum()
        n_bg = height * width - n_fg
        fg_pts = []
        bg_pts = []
        fg_pts.append(torch.sum(fts * mask[None, ...], dim=(-2, -1)) / (n_fg + 1e-5))
        bg_pts.append(torch.sum(fts * (1 - mask[None, ...]), dim=(-2, -1)) / (n_bg + 1e-5))
        patch_fts = F.unfold(fts, kernel_size=kernel_size, stride=stride).permute(0, 2, 1).view(1, -1, c, kernel_h, kernel_w)
        patch_mask = F.unfold(mask.unsqueeze(1), kernel_size=kernel_size, stride=stride).permute(0, 2, 1).view(1, -1, kernel_h, kernel_w).unsqueeze(2)
        bg_patch_mask = 1 - patch_mask
        if flag == 1:
            fg_indx = torch.sum(patch_mask, dim=(-2, -1)).view(-1) >= min(kernel_h*kernel_w*self.rate, n_fg/100)
            fg_protos = torch.sum(patch_fts[:, fg_indx, :, :, :] * patch_mask[:, fg_indx, :, :, :], dim=(-2, -1)) \
                        / (patch_mask[:, fg_indx, :, :, :].sum(dim=(-2, -1)) + 1e-5)
            if fg_protos.numel() == 0:
                fg_protos = self.get_pts(fts, mask)
            else:
                fg_protos = fg_protos.squeeze(0)
                if len(fg_protos) > 150:
                    fg_protos = fg_protos[random.sample(range(len(fg_protos)), 150)]
            fg_pts.append(fg_protos)

        bg_indx = torch.sum(bg_patch_mask, dim=(-2, -1)).view(-1) >= min(kernel_h*kernel_w, n_bg/100)
        bg_protos = torch.sum(patch_fts[:, bg_indx, :, :, :] * bg_patch_mask[:, bg_indx, :, :, :], dim=(-2, -1)) \
                    / (bg_patch_mask[:, bg_indx, :, :, :].sum(dim=(-2, -1)) + 1e-5)
        if bg_protos.numel() == 0:
            bg_protos = self.get_pts(fts, 1-mask)
        else:
            bg_protos = bg_protos.squeeze(0)
            if len(bg_protos) > 150:
                bg_protos = bg_protos[random.sample(range(len(bg_protos)), 150)]
        bg_pts.append(bg_protos)

        fg_pts = torch.cat(fg_pts, dim=0)
        bg_pts = torch.cat(bg_pts, dim=0)

        return [bg_pts, fg_pts]

    def match_pts(self, pts, qry_fts, qry_mask, thresh_pred):
        """
        Args:
            pts: expect shape: (n+1) x C
            qry_fts: expect shape: (N_q, 512, 64, 64)
            qry_mask: expect shape: (N_q, 1, 64, 64)
        """
        n, c, h, w = qry_fts.shape

        res = []
        for i in range(n):
            qry_ft = qry_fts[i].unsqueeze(0)
            qry_proto = self.get_masked_fts(qry_ft, qry_mask[i])
            pts = pts + torch.sigmoid(thresh_pred[i]) * qry_proto

            prototype = pts[0].unsqueeze(0)
            pts = pts[1:]     # regional_pts
            pts_ = F.normalize(pts, dim=-1)
            prototype = F.normalize(prototype, dim=-1)
            # match pts for qrt_fts
            fts_ = qry_ft.permute(0, 2, 3, 1)
            fts_ = F.normalize(fts_, dim=-1)

            one_sim = torch.matmul(fts_, prototype.transpose(0, 1)).permute(0, 3, 1, 2)
            # BASE
            sim = torch.matmul(fts_, pts_.transpose(0, 1)).permute(3, 0, 1, 2)
            sigma = qry_ft - pts.unsqueeze(-1).unsqueeze(-1)
            sigma = torch.sqrt(torch.mean(torch.square(sigma), dim=1))
            sigma = 2 / (1 + torch.exp(sigma * self.y))  # (n, h, w)

            sim += sigma.unsqueeze(1) * self.a

            sim_map = torch.softmax(sim*(1+self.scaler*torch.sigmoid(self.thresh_pred1[i, [0]])), dim=0)
            sim_0 = torch.sum(sim*sim_map, dim=0).unsqueeze(0)
            sim_1 = torch.sum(sim, dim=0).unsqueeze(0)
            matched_pts = pts.unsqueeze(2).unsqueeze(3) * sim_map
            matched_pts = torch.sum(matched_pts, dim=0).unsqueeze(0)

            matched_fts = torch.cat([qry_ft, matched_pts, sim_0, sim_1, one_sim], dim=1)
            res.append(matched_fts)

        res = torch.cat(res, dim=0)

        return res

    def get_qry_masks(self, pts, qry_fts, t):
        n = qry_fts.shape[0]
        pts_ = F.normalize(pts, dim=-1)
        qry_masks = []
        for i in range(n):
            qry_ft = qry_fts[i].unsqueeze(0)
            fts_ = qry_ft.permute(0, 2, 3, 1)
            fts_ = F.normalize(fts_, dim=-1)
            # BASE
            sim = torch.matmul(fts_, pts_.transpose(0, 1)).permute(3, 0, 1, 2)
            sigma = qry_ft - pts.unsqueeze(-1).unsqueeze(-1)
            sigma = torch.sqrt(torch.mean(torch.square(sigma), dim=1))
            sigma = 2 / (1 + torch.exp(sigma * self.y))  # (n, h, w)
            sim += sigma.unsqueeze(1) * self.a

            # generate prior mask
            sim_map = torch.softmax(sim*(1+self.scaler*torch.sigmoid(self.thresh_pred1[i, [0]])), dim=0)
            sim_soft = torch.sum(sim*sim_map, dim=0) * self.scaler
            pred_mask = torch.sigmoid((sim_soft - t))

            pred_mask = torch.cat([1 - pred_mask, pred_mask], dim=0)
            pred_mask = pred_mask.argmax(dim=0, keepdim=True).unsqueeze(1)

            qry_masks.append(pred_mask)

        qry_masks = torch.cat(qry_masks, dim=0).float()

        return qry_masks

    def get_qry_fts(self, qry_fts, sup_fts, mask, qry_masks):
        qry_fts_att = []
        mask = F.interpolate(mask.unsqueeze(0), size=sup_fts.shape[-2:], mode='bilinear', align_corners=True).squeeze(0)
        mask_sup_fts = sup_fts * mask
        for i in range(len(qry_fts)):
            mask_qry_ft = qry_fts[[i]] * qry_masks[i]
            # SRIA
            qry_ft_self = self.IA1(qry_fts[[i]], mask_qry_ft, qry_fts[[i]])
            # CRIA
            qry_ft_cross = self.IA2(qry_fts[[i]], mask_sup_fts, sup_fts)
            # mine target regions
            qry_fts_att.append((qry_ft_self * 0.5 + qry_ft_cross * self.beta1) / (0.5 + self.beta1))

        qry_fts_att = torch.cat(qry_fts_att, dim=0)
        return qry_fts_att

    def get_qry_bg_fts(self, qry_fts, sup_fts, mask, qry_masks):
        qry_fts_att = []
        mask = F.interpolate(mask.unsqueeze(0), size=sup_fts.shape[-2:], mode='bilinear', align_corners=True).squeeze(0)
        mask_sup_fts = sup_fts * (1-mask)
        for i in range(len(qry_fts)):
            mask_qry_ft = qry_fts[[i]] * qry_masks[i]
            # SRIA
            qry_ft_self = self.IA1(qry_fts[[i]], mask_qry_ft, qry_fts[[i]])
            # CRIA
            qry_ft_cross = self.IA2(qry_fts[[i]], mask_sup_fts, sup_fts)
            # mine target regions
            qry_fts_att.append((qry_ft_self * 0.5 + qry_ft_cross * self.beta2) / (0.5 + self.beta2))

        qry_fts_att = torch.cat(qry_fts_att, dim=0)
        return qry_fts_att
