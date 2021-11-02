#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader
from dataloader import TestDataset, ClassDataset
import utils.hyperbolic_utils as hyperbolic
#from termcolor import colored
from utils.math_utils import arsinh, arcosh
from utils import manifolds
from operator import itemgetter
import os
import math
import random
from itertools import combinations

class KGEModel(nn.Module):
    def __init__(self, model_name, nentity, nrelation, hidden_dim, gamma, 
                 double_entity_embedding=False, double_relation_embedding=False,
                 rid2cid=None, args=None):
        super(KGEModel, self).__init__()
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0
        self.rid2cid = rid2cid
        self.args = args
        self.gamma = nn.Parameter(
            torch.Tensor([gamma]), 
            requires_grad=False
        )
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
            requires_grad=False
        )
        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()
        self.softplus = nn.Softplus()

        if model_name in ['RotC', 'ConE']:
            self.embedding_range = nn.Parameter(
                torch.Tensor([1e-2]),
                requires_grad=False
            )
            dummy_index = nentity - 1
            self.dummy_node = dummy_index

        self.entity_dim = hidden_dim*2 if double_entity_embedding else hidden_dim
        self.relation_dim = hidden_dim*2 if double_relation_embedding else hidden_dim

        if model_name in ['RotC', 'ConE']:
            if args.learnable_curvature:
                self.curvature = nn.Parameter(torch.zeros(1, ))
            else:
                self.curvature = 1
            self.relation_dim += self.entity_dim
            # extra dimension for radius around the center
            self.entity_dim += hidden_dim
            self.relation_dim += 1
            K = 0.1
            self.K = K
            self.EPS = 1e-3
            self.inner_radius = 2 * K / (1 + np.sqrt(1 + 4 * K * K))
            self.manifold = manifolds.PoincareManifold(K=K)

        self.model_name = model_name

        self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim))

        nn.init.uniform_(
            tensor=self.entity_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )
        self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
        nn.init.uniform_(
            tensor=self.relation_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )
        if model_name in ['RotE', 'RotC']:
            self.entity_embedding.data[:, 2*hidden_dim:] = gamma * torch.ones(nentity, hidden_dim)

        # Initialize outside of the K ball, but inside the unit ball.
        if model_name in ['ConE']:

            self.dummy_node = nentity - 1
            self.dummy_relation = nrelation - 1

            # Load the pretrained poincare embedding as intialization
            init_path = args.pretrained
            ckpt = torch.load(init_path)
            self.load_state_dict(ckpt['model_state_dict'])
            print('=> Init checkpoint from %s' % init_path)
            self.gamma = nn.Parameter(
                torch.Tensor([gamma]), 
                requires_grad=False
            )
            if model_name == 'ConE':
                resc_vecs = 0.5
                print('=> RotC embedding factor: ', resc_vecs)

            with torch.no_grad():

                self.entity_embedding.data *= resc_vecs
                self.rel_dim = args.fix_att # each 1-N relation enforce cone restriction on rel_dim dimensions
                self.rel2dim = dict()
                dims = [i for i in range(hidden_dim)]
                disjoint = False
                if disjoint:
                    # disjoint subspaces
                    current = 0
                    count = 0
                    for rel in rid2cid.keys():
                        if rid2cid[rel] != 0:
                            count += 1
                    self.rel_dim = int(hidden_dim / count) # take up all the dimensions
                    for rel in rid2cid.keys():
                        if rid2cid[rel] != 0:
                            self.rel2dim[rel] = dims[current: current + self.rel_dim]
                            current += self.rel_dim
                else:
                    # overlapping subspaces
                    for rel in rid2cid.keys():
                        if rid2cid[rel] != 0:
                            self.rel2dim[rel] = random.sample(dims, self.rel_dim)
                # print(self.rel2dim)
                if self.args.fix_att:
                    mask = torch.ones(1, self.rel_dim)
                    self.relation_mask_embedding = nn.Parameter(torch.zeros(self.relation_embedding.data.size(0), hidden_dim), requires_grad=False)
                    for rel in self.rel2dim.keys():
                        self.relation_mask_embedding[rel, self.rel2dim[rel]] = mask
                nn.init.uniform_(tensor=self.relation_embedding[:, hidden_dim:2*hidden_dim], a = 0.5, b = 1)
                nn.init.uniform_(tensor=self.relation_embedding[:, 2*hidden_dim:3*hidden_dim], a = -0.1, b = 0.1)                        

        if model_name == 'pRotatE':
            self.modulus = nn.Parameter(torch.Tensor([[0.5 * self.embedding_range.item()]]))

        #Do not forget to modify this line when you add a new model in the "forward" function
        if model_name not in ['TransE', 'DistMult', 'ComplEx', 'RotatE', '2tRotatE', '4tRotatE', 'pRotatE', \
                              'RotC', 'ConE']:
            raise ValueError('model %s not supported' % model_name)

        if model_name == 'RotatE' and (not double_entity_embedding or double_relation_embedding):
            raise ValueError('RotatE should use --double_entity_embedding')

        if model_name == 'tRotatE' and (not double_entity_embedding or double_relation_embedding):
            raise ValueError('tRotatE should use --double_entity_embedding')

        if model_name == 'ComplEx' and (not double_entity_embedding or not double_relation_embedding):
            raise ValueError('ComplEx should use --double_entity_embedding and --double_relation_embedding')

        print('Embedding range: %.5f' % self.embedding_range.data)

    def forward(self, sample, mode='single', degree=None):
        '''
        Forward function that calculate the score of a batch of triples.
        In the 'single' mode, sample is a batch of triple.
        In the 'head-batch' or 'tail-batch' mode, sample consists two part.
        The first part is usually the positive sample.
        And the second part is the entities in the negative samples.
        Because negative samples and positive samples usually share two elements 
        in their triple ((head, relation) or (relation, tail)).
        '''

        if mode == 'single':
            batch_size, negative_sample_size = sample.size(0), 1
            
            head = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=sample[:,0]
            ).unsqueeze(1)
            
            relation = torch.index_select(
                self.relation_embedding, 
                dim=0, 
                index=sample[:,1]
            ).unsqueeze(1)
            
            tail = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=sample[:,2]
            ).unsqueeze(1)

            if sample.shape[1] == 4:
                relation_category = sample[:, 3]
                relation = (relation, relation_category)
            
            if self.args.fix_att:
                relation_mask = torch.index_select(
                    self.relation_mask_embedding, 
                    dim=0, 
                    index=sample[:,1]
                ).unsqueeze(1)
                relation = (relation, relation_mask)
            
        elif mode == 'head-batch':
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)
            
            head = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)
            
            relation = torch.index_select(
                self.relation_embedding, 
                dim=0, 
                index=tail_part[:, 1]
            ).unsqueeze(1)
            
            tail = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=tail_part[:, 2]
            ).unsqueeze(1)

            if tail_part.shape[1] == 4:
                relation_category = tail_part[:, 3]
                relation = (relation, relation_category)

            if self.args.fix_att:
                relation_mask = torch.index_select(
                    self.relation_mask_embedding, 
                    dim=0, 
                    index=tail_part[:, 1]
                ).unsqueeze(1)
                relation = (relation, relation_mask)

            
        elif mode == 'tail-batch':
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)

            head = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=head_part[:, 0]
            ).unsqueeze(1)
            
            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)


            tail = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

            if head_part.shape[1] == 4:
                relation_category = head_part[:, 3]
                relation = (relation, relation_category)
            
            if self.args.fix_att:
                relation_mask = torch.index_select(
                    self.relation_mask_embedding, 
                    dim=0, 
                    index=head_part[:, 1]
                ).unsqueeze(1)
                relation = (relation, relation_mask)
            
        else:
            raise ValueError('mode %s not supported' % mode)
            
        model_func = {
            'TransE': self.TransE,
            'DistMult': self.DistMult,
            'ComplEx': self.ComplEx,
            'RotatE': self.RotatE,
            'RotC': self.RotC,
            'ConE': self.ConE,
        }
        
        if self.model_name in model_func:
            score = model_func[self.model_name](head, relation, tail, mode)
        else:
            raise ValueError('model %s not supported' % self.model_name)
        
        return score
    
    def TransE(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail

        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score

    def DistMult(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head * (relation * tail)
        else:
            score = (head * relation) * tail

        score = score.sum(dim = 2)
        return score

    def ComplEx(self, head, relation, tail, mode):
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_relation, im_relation = torch.chunk(relation, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            score = re_head * re_score + im_head * im_score
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            score = re_score * re_tail + im_score * im_tail

        score = score.sum(dim = 2)
        return score

    def RotatE(self, head, relation, tail, mode):
        pi = 3.14159265358979323846
        
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        #Make phases of relations uniformly distributed in [-pi, pi]
        phase_relation = relation/(self.embedding_range.item()/pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim = 0)
        score = score.norm(dim = 0)

        score = self.gamma.item() - score.sum(dim = 2) 
        return score

    def RotC(self, head, relation, tail, mode): # a number of $dim$ 2D hyperbolic planes
        if isinstance(relation, tuple):
            relation_category = relation[1]
            relation = relation[0]

        pi = 3.14159265358979323846

        # project to hyperbolic manifold
        if self.args.learnable_curvature:
            c = self.softplus(self.curvature)
        else:
            c = self.curvature
        head, bh = head.split([int(2*self.entity_dim/3), int(self.entity_dim/3)], dim=2)
        tail, bt = tail.split([int(2*self.entity_dim/3), int(self.entity_dim/3)], dim=2)
        
        batch_size = head.size()[0]
        dim = int(head.size()[2] / 2)
        head = head.view(batch_size, -1, 2, dim).transpose(2, 3)
        tail = tail.view(batch_size, -1, 2, dim).transpose(2, 3)
        
        head = hyperbolic.proj(hyperbolic.expmap0(head, c), c=c)
        tail = hyperbolic.proj(hyperbolic.expmap0(tail, c), c=c)

        head_ = (head.transpose(2, 3)).contiguous().view(batch_size, -1, 2*dim)
        re_head, im_head = torch.chunk(head_, 2, dim=2)

        phase_relation, translation, _ = relation.split([int((self.entity_dim)/3.), int(2*(self.entity_dim)/3.), 1], dim=2)
        phase_relation = phase_relation / (self.embedding_range.item() / pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        re_rot = re_head * re_relation - im_head * im_relation
        im_rot = re_head * im_relation + im_head * re_relation
        res = torch.cat([re_rot, im_rot], dim=2)
        res = res.view(batch_size, -1, 2, dim).transpose(2, 3)
        
        score = hyperbolic.sqdist(res, tail, c).squeeze(3)
        if self.args.sum_loss:
            score = (bh + bt) - score.sum(dim=2).unsqueeze(dim=-1)
            score = score.mean(dim=2)
        else:
            score = bh + bt - score
            score = score.mean(dim=2)


        return score
    
    def ConE(self, head, relation, tail, mode):
        if self.args.fix_att:
            relation, relation_mask = relation
        if isinstance(relation, tuple):
            relation_category = relation[1]
            relation = relation[0]

        pi = 3.14159265358979323846
        # project to hyperbolic manifold
        if self.args.learnable_curvature:
            c = self.softplus(self.curvature)
        else:
            c = self.curvature
        head, bh = head.split([int(2 * self.entity_dim / 3), int(self.entity_dim / 3)], dim=2)
        tail, bt = tail.split([int(2 * self.entity_dim / 3), int(self.entity_dim / 3)], dim=2)
        batch_size = head.size()[0]
        dim = int(head.size()[2] / 2)
        head = head.view(batch_size, -1, 2, dim).transpose(2, 3)
        tail = tail.view(batch_size, -1, 2, dim).transpose(2, 3)
        head = hyperbolic.expmap0(head, c)
        tail = hyperbolic.expmap0(tail, c)

        head_rot = (head.transpose(2, 3)).view(batch_size, -1, 2*dim)
        tail_rot = (tail.transpose(2, 3)).view(batch_size, -1, 2*dim)
        re_head, im_head = torch.chunk(head_rot, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail_rot, 2, dim=2)

        phase_relation, translation, _ = relation.split([int((self.entity_dim)/3.), int(2*(self.entity_dim)/3.), 1], dim=2)
        phase_relation = phase_relation / (self.embedding_range.item() / pi)
        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)
        re_rot = re_head * re_relation - im_head * im_relation
        im_rot = re_head * im_relation + im_head * re_relation
        rot = torch.cat([re_rot, im_rot], dim=2)
        rot = rot.view(batch_size, -1, 2, dim).transpose(2, 3)

        one_one_mask = (relation_category == 0).view(-1, 1, 1)
        one_many_mask = (relation_category == 1).view(-1, 1, 1)
        many_one_mask = (relation_category == 2).view(-1, 1, 1)

        # res_1 by rotation transformation from h to t
        # res_2 by restricted rotation transformation from h to t (hyponym type)
        # res_2 by restricted rotation transformation from t to h (hypernym type)
        
        res_1 = rot
        res_2 = self.cone_rotate(head, relation, tail, bh, bt)
        res_3 = self.cone_rotate(tail, relation, head, bt, bh)
        score_1 = hyperbolic.sqdist(res_1, tail, c).squeeze(3)
        score_2 = hyperbolic.sqdist(res_2, tail, c).squeeze(3)
        score_3 = hyperbolic.sqdist(res_3, head, c).squeeze(3)

        if self.args.fix_att:
            score_2 = score_2 * relation_mask + score_1 * (1 - relation_mask)
            score_3 = score_3 * relation_mask + score_1 * (1 - relation_mask)
        
        score = score_1 * one_one_mask.float() + score_2 * one_many_mask.float() + score_3 * many_one_mask.float()
        if self.args.sum_loss:
            score = (bh + bt) - score.sum(dim=2).unsqueeze(dim=-1)
            score = score.mean(dim=2)
        else:
            score = bh + bt - score
            score = score.mean(dim=2)

        if self.args.cone_penalty:
            energy_cones_1 = self.score_cones(head, tail, bh)
            energy_cones_2 = self.score_cones(tail, head, bt)
            energy_cones = energy_cones_1 * one_many_mask.float() + energy_cones_2 * many_one_mask.float()
            if self.args.fix_att:
                score = (score, (energy_cones * relation_mask).mean(dim=-1))
            else:
                score = (score, energy_cones.mean(dim=-1))
        return score
    
    def cone_rotate(self, head, relation, tail, bh, bt): # do restricted rotation (for 1-N edges)
        pi = 3.14159265358979323846
        batch_size = head.size()[0]
        dim = int(head.size()[2])
        phase_relation, scale, _ = relation.split([int((self.entity_dim)/3.), int((self.entity_dim)/3.), int((self.entity_dim)/3.)+1], dim=2)
        if self.args.learnable_curvature:
            c = self.softplus(self.curvature)
        else:
            c = self.curvature
        # 1. calculate the scale to go further on the radius (in tangent space)
        head_unit = 0.1 * head
        head_bar = hyperbolic.logmap(head, head_unit, c)
        head_scale = scale.unsqueeze(dim=3).abs() * head_bar / head_bar.norm(dim=-1).unsqueeze(dim=-1)
        head_scale = (head_scale.transpose(2, 3)).contiguous().view(batch_size, -1, 2*dim)
        re_head, im_head = torch.chunk(head_scale, 2, dim=2)
        # 2. compute restricted angle then rotate
        aperture = self.manifold.half_aperture(head)
        phase_relation = self.tanh(phase_relation) * aperture
        re_relation = torch.cos(pi - phase_relation)
        im_relation = torch.sin(pi - phase_relation)
        re_rot = re_head * re_relation - im_head * im_relation
        im_rot = re_head * im_relation + im_head * re_relation
        rot_head = torch.cat([re_rot, im_rot], dim=2)
        # 3. map back to poincare ball
        rot_head = rot_head.view(batch_size, -1, 2, dim).transpose(2, 3)
        res = hyperbolic.proj(hyperbolic.expmap(rot_head, head, c), c=c)
        return res

    def score_cones(self, x, y, bx, att=None):
        batch_size = x.size()[0]
        energy = self.manifold.angle_at_u(x, y) - self.manifold.half_aperture(x)
        energy = energy.clamp(min = 0)
        return energy

    def prob_cones(self, x, y):
        energy = 1 - self.manifold.angle_at_u(x, y) / self.manifold.half_aperture(x)
        return energy.clamp(min=0)

    def score_descendant(self, head, rel, tail, category):
        head = torch.index_select(
            self.entity_embedding, 
            dim=0, 
            index=head[:,0]
        )
        tail = torch.index_select(
            self.entity_embedding, 
            dim=0, 
            index=tail[:,0]
        )
        relation = torch.index_select(
            self.relation_embedding, 
            dim=0, 
            index=rel[:,0]
        )
        if self.args.fix_att:
            relation_mask = torch.index_select(
                self.relation_mask_embedding, 
                dim=0, 
                index=rel[:,0]
            )
        if self.args.learnable_curvature:
            c = self.softplus(self.curvature)
        else:
            c = self.curvature
        head, bh = head.split([int(2*self.entity_dim/3), int(self.entity_dim/3)], dim=1)
        tail, bt = tail.split([int(2*self.entity_dim/3), int(self.entity_dim/3)], dim=1)
        head = hyperbolic.proj(hyperbolic.expmap0(head, c), c=c)
        tail = hyperbolic.proj(hyperbolic.expmap0(tail, c), c=c)
        batch_size = head.size()[0]
        dim = int(head.size()[1] / 2)
        head = head.view(batch_size, 2, dim).transpose(1, 2)
        tail = tail.view(batch_size, 2, dim).transpose(1, 2)
        energy_1 = self.manifold.half_aperture(head) - self.manifold.angle_at_u(head, tail)
        energy_2 = self.manifold.half_aperture(tail) - self.manifold.angle_at_u(tail, head)
        if self.args.fix_att:
            energy_1 = (energy_1 * relation_mask).mean(dim = 1).view(-1, 1)
            energy_2 = (energy_2 * relation_mask).mean(dim = 1).view(-1, 1)
        else:
            energy_1 = energy_1.mean(dim = 1).view(-1, 1)
            energy_2 = energy_2.mean(dim = 1).view(-1, 1)
        one_many_mask = (category == 1)
        many_one_mask = (category == 2)
        score = energy_1 * one_many_mask.float() + energy_2 * many_one_mask.float()
        return score
    
    def evaluate_lca_queries(self, lca_query, entity_embedding, half_aperture, args):
        p, q, lca, rel = lca_query
        hit_1 = 0
        hit_3 = 0
        hit_10 = 0
        hit = 0
        batch_size = p.size(0)
        c = 1
        if self.args.fix_att:
            relation_mask = torch.index_select(
                self.relation_mask_embedding, 
                dim=0, 
                index=rel[:,0]
            )
        p = torch.index_select(
            entity_embedding, 
            dim=0, 
            index=p[:, 0]
        )
        q = torch.index_select(
            entity_embedding, 
            dim=0, 
            index=q[:, 0]
        )
        rel = torch.index_select(
            self.relation_embedding, 
            dim=0, 
            index=rel[:, 0]
        )
        entity_embedding = entity_embedding.unsqueeze(dim=0)
        p = p.unsqueeze(dim=1)
        q = q.unsqueeze(dim=1)
        rel = rel.unsqueeze(dim=1)
        if args.model == 'ConE':
            threshold = 0.0
            score_p = self.manifold.angle_at_u(entity_embedding, p) - half_aperture
            score_q = self.manifold.angle_at_u(entity_embedding, q) - half_aperture
            score = score_p.clamp(min=0.) + score_q.clamp(min=0.)
            if self.args.fix_att:
                score = (score * relation_mask.unsqueeze(dim=1)).mean(dim = -1)
            entity_embedding = entity_embedding.squeeze(dim=0)
            for i in range(batch_size):
                score_i = score[i]
                argsort = torch.argsort(score_i).tolist()
                one = [argsort[0]]
                three = argsort[0:3]
                ten = argsort[0:10]
                if lca[i].item() in one:
                    hit_1 += 1
                if lca[i].item() in three:
                    hit_3 += 1
                if lca[i].item() in ten:
                    hit_10 += 1
        return hit_1, hit_3, hit_10, hit

    @staticmethod
    def train_step(model, optimizer, train_iterator, args, step, viable_neg):
        '''
        A single train step. Apply back-propation and return the loss
        '''
        model.train()
        optimizer.zero_grad()

        if args.train_with_degree:
            positive_sample, negative_sample, subsampling_weight, mode, degree = next(train_iterator)
        else:
            degree = None
            positive_sample, negative_sample, subsampling_weight, mode = next(train_iterator)

        if args.cuda:
            positive_sample = positive_sample.cuda()
            if isinstance(negative_sample, list):
                negative_sample = [negative_sample[0].cuda(), negative_sample[1].cuda()]
            else:
                negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()
            degree = None if degree is None else degree.cuda()

        if args.cone_penalty:
            negative_score, negative_cone_score = model((positive_sample, negative_sample), mode=mode, degree=degree)
        else:
            negative_score = model((positive_sample, negative_sample), mode=mode, degree=degree)
        if args.negative_adversarial_sampling:
            #In self-adversarial sampling, we do not apply back-propagation on the sampling weight
            negative_score = (F.softmax(negative_score * args.adversarial_temperature, dim = 1).detach()
                                * F.logsigmoid(-negative_score)).sum(dim = 1)
        else:
            negative_score = F.logsigmoid(-negative_score).sum(dim = 1)

        if args.cone_penalty:
            positive_score, positive_cone_score = model(positive_sample, degree=degree)
        else:
            positive_score = model(positive_sample, degree=degree)

        positive_score = F.logsigmoid(positive_score).squeeze(dim = 1)
        if args.uni_weight:
            positive_sample_loss = - positive_score.mean()
            negative_sample_loss = - negative_score.mean()
        else:
            positive_sample_loss = - (subsampling_weight * positive_score).sum()/subsampling_weight.sum()
            negative_sample_loss = - (subsampling_weight * negative_score).sum()/subsampling_weight.sum()

        loss = (positive_sample_loss + negative_sample_loss) / 2
        
        if args.cone_penalty:
            positive_cone_loss = positive_cone_score.mean()
            negative_cone_loss = - negative_cone_score.mean(dim=1).mean()
            loss = loss + args.w * positive_cone_loss

        if args.regularization != 0.0:
            #Use L3 regularization for ComplEx and DistMult
            regularization = args.regularization * (
                model.entity_embedding.norm(p = 3)**3 +
                model.relation_embedding.norm(p = 3).norm(p = 3)**3
            )
            loss = loss + regularization
            regularization_log = {'regularization': regularization.item()}
        else:
            regularization_log = {}

        loss.backward()

        optimizer.step()

        if args.cone_penalty:
            log = {
                **regularization_log,
                'positive_sample_loss': positive_sample_loss.item(),
                'negative_sample_loss': negative_sample_loss.item(),
                'angle_loss': positive_cone_loss.item(),
                'loss': loss.item()
            }
        else:
            log = {
                **regularization_log,
                'positive_sample_loss': positive_sample_loss.item(),
                'negative_sample_loss': negative_sample_loss.item(),
                'loss': loss.item()
            }

        return log


    @staticmethod
    def test_step(model, test_triples, class_test_triples_list, lca_test_triples, all_true_triples, args, relation_category=False, degree=None):
        '''
        Evaluate the model on test or valid datasets
        '''

        model.eval()

        if args.countries:
            #Countries S* datasets are evaluated on AUC-PR
            #Process test data for AUC-PR evaluation
            sample = list()
            y_true  = list()
            for head, relation, tail in test_triples:
                for candidate_region in args.regions:
                    y_true.append(1 if candidate_region == tail else 0)
                    sample.append((head, relation, candidate_region))

            sample = torch.LongTensor(sample)
            if args.cuda:
                sample = sample.cuda()

            with torch.no_grad():
                y_score = model(sample).squeeze(1).cpu().numpy()

            y_true = np.array(y_true)

            #average_precision_score is the same as auc_pr
            auc_pr = average_precision_score(y_true, y_score)

            metrics = {'auc_pr': auc_pr}

        else:
            #Otherwise use standard (filtered) MRR, MR, HITS@1, HITS@3, and HITS@10 metrics
            #Prepare dataloader for evaluation
            test_dataloader_head = DataLoader(
                TestDataset(
                    test_triples,
                    all_true_triples,
                    args.nentity,
                    args.nrelation,
                    'head-batch',
                    degree
                ),
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num//2), # max(1, args.cpu_num//8),
                collate_fn=TestDataset.collate_fn
            )

            test_dataloader_tail = DataLoader(
                TestDataset(
                    test_triples,
                    all_true_triples,
                    args.nentity,
                    args.nrelation,
                    'tail-batch',
                    degree
                ),
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num//2),
                collate_fn=TestDataset.collate_fn
            )
            logs = []
            metrics = {}
            if args.do_lca: # LCA prediction task
                test_dataloader_lca = DataLoader(
                    ClassDataset(
                        lca_test_triples
                    ),
                    batch_size=args.test_batch_size,
                    num_workers=max(1, args.cpu_num//2),
                    collate_fn=ClassDataset.collate_fn
                )
                c = 1
                entity_embedding = model.entity_embedding
                half_aperture = None
                if args.model == 'ConE':
                    entity_embedding, be = entity_embedding.split([int(2 * model.entity_dim / 3), int(model.entity_dim / 3)], dim=1)
                    dim = int(entity_embedding.size(1) / 2)
                    entity_embedding = entity_embedding.view(-1, 2, dim).transpose(1, 2)
                    entity_embedding = hyperbolic.expmap0(entity_embedding, c)
                    half_aperture = model.manifold.half_aperture(entity_embedding)
                    entity_embedding = entity_embedding.cuda()
                    half_aperture = half_aperture.cuda()
                hit_1 = 0
                hit_3 = 0
                hit_10 = 0
                count = 0
                with torch.no_grad():
                    for data in test_dataloader_lca:
                        p, q, lca, rel = data
                        if args.cuda:
                            p = p.cuda().unsqueeze(1)
                            q = q.cuda().unsqueeze(1)
                            lca = lca.cuda().unsqueeze(1)
                            rel = rel.cuda().unsqueeze(1)
                        lca_query = (p, q, lca, rel)
                        hit1, hit3, hit10, hitt = model.evaluate_lca_queries(lca_query, entity_embedding, half_aperture, args)
                        hit_1 += hit1
                        hit_3 += hit3
                        hit_10 += hit10
                        count += data[0].size(0)
                hit_1 /= count
                hit_3 /= count
                hit_10 /= count
                metrics['LCA hit@1'] = hit_1
                metrics['LCA hit@3'] = hit_3
                metrics['LCA hit@10'] = hit_10
            if args.do_classification: # Ancstor-descendant classification task
                modes = ['easy', 'medium', 'hard']
                for i in range(3): # easy, medium, hard modes
                    class_test_triples = class_test_triples_list[i]
                    submit = None
                    ground_truth = None
                    test_dataloader_class = DataLoader(
                        ClassDataset(
                            class_test_triples
                        ),
                        batch_size=512 * args.negative_sample_size,
                        num_workers=1,
                        collate_fn=ClassDataset.collate_fn
                    )
                    with torch.no_grad():
                        for data in test_dataloader_class:
                            if args.do_test_relation_category:
                                h, r, t, l, c = data
                            else:
                                h, r, t, l = data
                            if args.cuda:
                                head = h.cuda().unsqueeze(1)
                                rel = r.cuda().unsqueeze(1)
                                tail = t.cuda().unsqueeze(1)
                                length = l.cuda().unsqueeze(1)
                            if args.do_test_relation_category:
                                category = c.cuda().unsqueeze(1)
                                sample = torch.cat((head, rel, tail, category), 1)
                            else:
                                sample = torch.cat((head, rel, tail), 1)
                            if args.model == 'ConE':
                                score = model.score_descendant(head, rel, tail, category)
                            else:
                                score = model(sample)
                            if isinstance(score, tuple):
                                score = -score[1].squeeze(2)
                            if submit is None:
                                submit = score
                                ground_truth = length
                            else:
                                submit = torch.cat((submit, score), 0)
                                ground_truth = torch.cat((ground_truth, length), 0)
                    submit = submit.squeeze().cpu().numpy()
                    ground_truth = ground_truth.squeeze().cpu().numpy()
                    total_all = len(submit)
                    total_current = 0.0
                    total_true = (ground_truth > 0).sum()
                    total_false = total_all - total_true
                    res = np.concatenate([ground_truth.reshape(-1,1), submit.reshape(-1,1)], axis = -1)
                    order = np.argsort(-submit)
                    res = res[order]
                    pre_rec = dict()
                    roc = dict()
                    total_current = 0.0
                    for index, [ans, score] in enumerate(res):
                        tpc = total_current # 0
                        fpc = (index - total_current) # 0
                        fnc = (total_true - total_current)
                        tpr = total_current / total_true # true positive rate
                        fpr = (index - total_current) / total_false # false positive rate
                        roc[fpr] = tpr
                        if (tpc + fpc) == 0:
                            precision = 1.0
                        else:
                            precision = tpc / (tpc + fpc)
                        recall = tpc / (tpc + fnc)
                        if recall not in pre_rec:
                            pre_rec[recall] = precision
                        if ans > 0.5:
                            total_current += 1.0
                    # interpolation
                    pre_rec_itp = dict()
                    itp = 0.000001
                    itp_current = 0.0
                    for rec, pre in pre_rec.items():
                        while rec > itp_current:
                            pre_rec_itp[itp_current] = pre
                            itp_current += itp
                    mAp = 0.0
                    for pre in pre_rec_itp.values():
                        mAp += pre * itp
                    roc_itp = dict()
                    itp = 0.000001
                    itp_current = 0.0
                    for fpr, tpr in roc.items():
                        while fpr > itp_current:
                            roc_itp[itp_current] = tpr
                            itp_current += itp
                    AUC = 0.0
                    for tpr in roc_itp.values():
                        AUC += tpr * itp
                    metrics[modes[i]+' mAp'] = mAp
                    metrics[modes[i]+' AUC'] = AUC
            
            test_dataset_list = [test_dataloader_head, test_dataloader_tail]
            hits = []

            step = 0
            total_steps = sum([len(dataset) for dataset in test_dataset_list])

            total_score = 0
            total_score_neg = 0
            total_size = 0
            triple_set = set(all_true_triples)
            with torch.no_grad():
                for test_dataset in test_dataset_list:
                    for data in test_dataset:
                        if args.train_with_degree:
                            positive_sample, negative_sample, filter_bias, mode, degree = data
                        else:
                            degree = None
                            positive_sample, negative_sample, filter_bias, mode = data
                        if args.cuda:
                            positive_sample = positive_sample.cuda()
                            negative_sample = negative_sample.cuda()
                            filter_bias = filter_bias.cuda()
                            degree = None if degree is None else degree.cuda()

                        batch_size = positive_sample.size(0)

                        if args.train_with_relation_category:
                            assert positive_sample.size(1) == 4
                            category = positive_sample[:, 3]
                            relation = positive_sample[:, 1]
                        else:
                            if relation_category:
                                assert positive_sample.size(1) == 4
                                category = positive_sample[:, 3]
                                relation = positive_sample[:, 1]
                        
                        score = model((positive_sample, negative_sample), mode, degree)
                        neg_score = score[1].squeeze().mean()
                        total_score_neg += neg_score

                        pos_score = model(positive_sample)
                        if args.cone_penalty:
                            score = score[0]

                        score += (filter_bias * 100)
                        argsort = torch.argsort(score, dim=1, descending=True)

                        if mode == 'head-batch':
                            positive_arg = positive_sample[:, 0]
                        elif mode == 'tail-batch':
                            positive_arg = positive_sample[:, 2]
                        else:
                            raise ValueError('mode %s not supported' % mode)

                        for i in range(batch_size):
                            # Notice that argsort is not ranking
                            ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                            assert ranking.size(0) == 1

                            # ranking + 1 is the true ranking used in evaluation metrics
                            ranking = 1 + ranking.item()
                            content = {
                                'MRR': 1.0/ranking,
                                'MR': float(ranking),
                                'HITS@1': 1.0 if ranking <= 1 else 0.0,
                                'HITS@3': 1.0 if ranking <= 3 else 0.0,
                                'HITS@10': 1.0 if ranking <= 10 else 0.0,
                            }
                            logs.append(content)

                        if step % args.test_log_steps == 0:
                            logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                        step += 1
            for metric in logs[0].keys():
                metrics[metric] = sum([log[metric] for log in logs])/len(logs)

        return metrics
