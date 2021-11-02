#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import logging
import os
import random

import numpy as np
import torch
import pickle
from torch.utils.data import DataLoader
from model import KGEModel

from dataloader import TrainDataset, TrainDatasetDualNegative
from dataloader import BidirectionalOneShotIterator, UnidirectionalOneShotIterator
from utils.rsgd import RiemannianSGD
from utils import manifolds
import time


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='train.py [<args>] [-h | --help]'
    )

    parser.add_argument('--cuda', action='store_true', help='use GPU')
    
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_valid', action='store_true')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('-dtc', '--do_test_relation_category', action='store_true')
    parser.add_argument('--train_with_relation_category', action='store_true')
    parser.add_argument('--train_with_degree', action='store_true')
    parser.add_argument('--select_negative_samples', action='store_true')
    parser.add_argument('--dual_negative_samples', action='store_true')
    parser.add_argument('--train_txt', default='train.txt', type=str)


    parser.add_argument('--evaluate_train', action='store_true', help='Evaluate on training data')
    
    parser.add_argument('--countries', action='store_true', help='Use Countries S1/S2/S3 datasets')
    parser.add_argument('--regions', type=int, nargs='+', default=None, 
                        help='Region Id for Countries S1/S2/S3 datasets, DO NOT MANUALLY SET')
    
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--model', default='TransE', type=str)
    parser.add_argument('-de', '--double_entity_embedding', action='store_true')
    parser.add_argument('-dr', '--double_relation_embedding', action='store_true')
    
    parser.add_argument('-n', '--negative_sample_size', default=128, type=int)
    parser.add_argument('-d', '--hidden_dim', default=500, type=int)
    parser.add_argument('-g', '--gamma', default=12.0, type=float)
    parser.add_argument('-adv', '--negative_adversarial_sampling', action='store_true')
    parser.add_argument('-a', '--adversarial_temperature', default=1.0, type=float)
    parser.add_argument('-b', '--batch_size', default=1024, type=int)
    parser.add_argument('-r', '--regularization', default=0.0, type=float)
    parser.add_argument('--test_batch_size', default=4, type=int, help='valid/test batch size')
    parser.add_argument('--uni_weight', action='store_true', 
                         help='Otherwise use subsampling weighting like in word2vec')
    
    parser.add_argument('-lr', '--learning_rate', default=0.0001, type=float)
    parser.add_argument('-cpu', '--cpu_num', default=10, type=int)
    parser.add_argument('-init', '--init_checkpoint', default=None, type=str)
    parser.add_argument('-ckpt', '--checkpoint_name', default=None, type=str)
    parser.add_argument('-save', '--save_path', default=None, type=str)
    parser.add_argument('-tb', '--tb_path', default=None, type=str, help='path to tensorboard log dir')
    parser.add_argument('--max_steps', default=100000, type=int)
    parser.add_argument('--lr_decay_epoch', default=None, type=str)
    parser.add_argument('--lr_decay_rate', default=0.2, type=float)

    
    parser.add_argument('--save_checkpoint_steps', default=10000, type=int)
    parser.add_argument('--valid_steps', default=10000, type=int)
    parser.add_argument('--log_steps', default=100, type=int, help='train log every xx steps')
    parser.add_argument('--test_log_steps', default=1000, type=int, help='valid/test log every xx steps')
    
    parser.add_argument('--nentity', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('--nrelation', type=int, default=0, help='DO NOT MANUALLY SET')

    parser.add_argument('--tail_batch_only', action='store_true',
                        help='use tail batch only for training')
    parser.add_argument('--head_batch_only', action='store_true',
                        help='use head batch only for training')
    # below are some important arguments for ConE model
    parser.add_argument('--do_classification', action='store_true', help='do ancestor-descendant prediction task')
    parser.add_argument('--do_lca', type=int, default=0, help='do lca prediction task')
    parser.add_argument('--learnable_curvature', action='store_true', help='use learnable curvature, otherwise fixed at -1')
    parser.add_argument('--cone_penalty', action='store_true', help='add angle loss')
    parser.add_argument('--fix_att', type=int, default=0, help='subspace dimension in ConE model')
    parser.add_argument('--w', type=float, default=0.0, help='weight for cone loss')
    parser.add_argument('--sum_loss', action='store_true', help='summation loss in each batch for faster SGD')
    parser.add_argument('--pretrained', default=None, type=str, help='path for ConE model to load pretrained RotC model')
    return parser.parse_args(args)

def override_config(args):
    '''
    Override model and data configuration
    '''
    
    with open(os.path.join(args.init_checkpoint, 'config.json'), 'r') as fjson:
        argparse_dict = json.load(fjson)
    
    args.countries = argparse_dict['countries']
    if args.data_path is None:
        args.data_path = argparse_dict['data_path']
    args.model = argparse_dict['model']
    args.double_entity_embedding = argparse_dict['double_entity_embedding']
    args.double_relation_embedding = argparse_dict['double_relation_embedding']
    args.hidden_dim = argparse_dict['hidden_dim']
    
def save_model(model, optimizer, save_variable_list, args, step):
    '''
    Save the parameters of the model and the optimizer,
    as well as some other variables such as step and learning_rate
    '''
    
    argparse_dict = vars(args)
    with open(os.path.join(args.save_path, 'config.json'), 'w') as fjson:
        json.dump(argparse_dict, fjson)

    if not os.path.exists(os.path.join(args.save_path, 'checkpoint')):
        os.mkdir(os.path.join(args.save_path, 'checkpoint'))

    torch.save({
        **save_variable_list,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        os.path.join(args.save_path, 'checkpoint', 'ckpt_%d' % step)
    )
    
    entity_embedding = model.entity_embedding.detach().cpu().numpy()
    np.save(
        os.path.join(args.save_path, 'entity_embedding'), 
        entity_embedding
    )
    
    relation_embedding = model.relation_embedding.detach().cpu().numpy()
    np.save(
        os.path.join(args.save_path, 'relation_embedding'), 
        relation_embedding
    )

def read_triple(file_path, entity2id, relation2id):
    '''
    Read triples and map them into ids.
    '''
    triples = []
    with open(file_path) as fin:
        for line in fin:
            h, r, t = line.strip().split('\t')
            triples.append((entity2id[h], relation2id[r], entity2id[t]))
    return triples

def read_class(file_path, entity2id):
    '''
    Read classification data and map them into ids. 
    format: head \t relation \t tail \t length
    '''
    triples = []
    with open(file_path) as fin:
        for line in fin:
            h, r, t, l = line.strip().split('\t')
            triples.append((int(h), int(r), int(t), int(l)))
    return triples

def read_lca(file_path, entity2id):
    '''
    Read lca data and map them into ids. 
    format: entity_p \t entity_q \t lca \t relation
    '''
    triples = []
    with open(file_path) as fin:
        for line in fin:
            p, q, lca, r = line.strip().split('\t')
            triples.append((int(p), int(q), int(lca), int(r)))
    return triples

def set_logger(args):
    '''
    Write logs to checkpoint and console
    '''

    if args.do_train:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'train.log')
    else:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'test.log')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def log_metrics(mode, step, metrics):
    '''
    Print the evaluation logs
    '''
    for metric in metrics:
        logging.info('%s %s at step %d: %f' % (mode, metric, step, metrics[metric]))

def main(args):
    if (not args.do_train) and (not args.do_valid) and (not args.do_test) and not (args.do_test_relation_category):
        raise ValueError('one of train/val/test mode must be choosed.')
    
    if args.init_checkpoint:
        override_config(args)
    elif args.data_path is None:
        raise ValueError('one of init_checkpoint/data_path must be choosed.')

    if args.do_train and args.save_path is None:
        raise ValueError('Where do you want to save your trained model?')

    if args.save_path and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    # Write logs to checkpoint and console
    set_logger(args)
    
    with open(os.path.join(args.data_path, 'entities.dict')) as fin:
        entity2id = dict()
        for line in fin:
            eid, entity = line.strip().split('\t')
            entity2id[entity] = int(eid)

    with open(os.path.join(args.data_path, 'relations.dict')) as fin:
        relation2id = dict()
        for line in fin:
            rid, relation = line.strip().split('\t')
            relation2id[relation] = int(rid)
    rid2cid = None
    if args.do_test_relation_category or args.train_with_relation_category:
        rid2cid = dict()
        category2id = {'1-1': 0, '1-M': 1, 'M-1': 2, 'M-M': 3, 'dummy': -1}

        with open(os.path.join(args.data_path, 'relation_category.txt')) as fin:
            for line in fin:
                relation, category = line.strip().split('\t')
                rid2cid[relation2id[relation]] = category2id[category]
    
    # Read regions for Countries S* datasets
    if args.countries:
        regions = list()
        with open(os.path.join(args.data_path, 'regions.list')) as fin:
            for line in fin:
                region = line.strip()
                regions.append(entity2id[region])
        args.regions = regions


    nentity = len(entity2id)
    nrelation = len(relation2id)
    
    args.nentity = nentity
    args.nrelation = nrelation
    args.relation2id = relation2id

    logging.info('Model: %s' % args.model)
    logging.info('Data Path: %s' % args.data_path)

    logging.info(' ')
    logging.info('#entity: %d' % nentity)
    logging.info('#relation: %d' % nrelation)
    
    train_triples = read_triple(os.path.join(args.data_path, args.train_txt), entity2id, relation2id)
    logging.info('#train: %d' % len(train_triples))
    valid_triples = read_triple(os.path.join(args.data_path, 'valid.txt'), entity2id, relation2id)
    logging.info('#valid: %d' % len(valid_triples))
    test_triples = read_triple(os.path.join(args.data_path, 'test.txt'), entity2id, relation2id)
    logging.info('#test: %d' % len(test_triples))
    
    class_test_triples_list = []
    class_test_triples_list_ = []
    class_test_triples = None
    lca_test_triples = None
    if args.do_classification:
        class_test_triples_easy = read_class(os.path.join(args.data_path, 'class_test_easy.txt'), entity2id)
        class_test_triples_medium = read_class(os.path.join(args.data_path, 'class_test_medium.txt'), entity2id)
        class_test_triples_hard = read_class(os.path.join(args.data_path, 'class_test_hard.txt'), entity2id)
        class_test_triples_list = [class_test_triples_easy, class_test_triples_medium, class_test_triples_hard]
    if args.do_lca > 0:
        lca_test_triples = read_lca(os.path.join(args.data_path, 'lca_test_'+str(args.do_lca)+'.txt'), entity2id)
        logging.info('#lca test X=%d: %d' % (args.do_lca, len(lca_test_triples)))
    if args.do_classification:
        if args.do_test_relation_category:
            for class_test_triples in class_test_triples_list:
                class_test_triples = [triple + (rid2cid[triple[1]],) for triple in class_test_triples]
                class_test_triples_list_.append(class_test_triples)
        else:
            class_test_triples_list_ = class_test_triples_list
    all_true_triples = train_triples + valid_triples + test_triples

    if args.do_test_relation_category:
        valid_triples = [triple + (rid2cid[triple[1]],) for triple in valid_triples]
        test_triples = [triple + (rid2cid[triple[1]],) for triple in test_triples]


    if args.train_with_relation_category:
        train_triples = [triple + (rid2cid[triple[1]],) for triple in train_triples]

    kge_model = KGEModel(
        model_name=args.model,
        nentity=nentity,
        nrelation=nrelation,
        hidden_dim=args.hidden_dim,
        gamma=args.gamma,
        double_entity_embedding=args.double_entity_embedding,
        double_relation_embedding=args.double_relation_embedding,
        rid2cid=rid2cid,
        args=args
    )

    logging.info(' ')
    logging.info('Model Parameter Configuration:')
    for name, param in kge_model.named_parameters():
        logging.info('Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))

    if args.cuda:
        kge_model = kge_model.cuda()

    if args.train_with_degree:
        with open(os.path.join(args.data_path, 'degree.pkl'), 'rb') as handle:
            degree = pickle.load(handle)
    else:
        degree = None

    if args.select_negative_samples or args.dual_negative_samples:
        with open(os.path.join(args.data_path, 'descendants.json')) as json_file:
            viable_neg = json.load(json_file)
    else:
        viable_neg = None
    
    if args.do_train:
        # Set training dataloader iterator

        dataset_func = TrainDatasetDualNegative if args.dual_negative_samples else TrainDataset

        train_dataloader_head = DataLoader(
            dataset_func(train_triples, nentity, nrelation, args.negative_sample_size, 'head-batch', degree, viable_neg),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=max(1, args.cpu_num//4),
            collate_fn=dataset_func.collate_fn
        )

        train_dataloader_tail = DataLoader(
            dataset_func(train_triples, nentity, nrelation, args.negative_sample_size, 'tail-batch', degree, viable_neg),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=max(1, args.cpu_num//4),
            collate_fn=dataset_func.collate_fn
        )
        if args.tail_batch_only:
            print('train on tail batch only')
            train_iterator = UnidirectionalOneShotIterator(train_dataloader_tail)
        elif args.head_batch_only:
            print('train on head batch only')
            train_iterator = UnidirectionalOneShotIterator(train_dataloader_head)
        else:
            train_iterator = BidirectionalOneShotIterator(train_dataloader_head, train_dataloader_tail)
        
        # Set training configuration
        current_learning_rate = args.learning_rate

        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, kge_model.parameters()),
            lr=current_learning_rate
        )

    if args.init_checkpoint:
        # Restore model from checkpoint directory
        logging.info('Loading checkpoint %s...' % args.init_checkpoint)
        if args.checkpoint_name is None:
            checkpoint = torch.load(os.path.join(args.init_checkpoint, 'checkpoint'))
        else:
            checkpoint = torch.load(os.path.join(args.init_checkpoint, 'checkpoint', args.checkpoint_name))
        init_step = checkpoint['step']
        kge_model.load_state_dict(checkpoint['model_state_dict'])

        if args.do_train:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            logging.info('Change adam learning_rate to %f' % current_learning_rate)

            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, kge_model.parameters()),
                lr=current_learning_rate
            )
    else:
        logging.info('Ramdomly Initializing %s Model...' % args.model)
        init_step = 0
    
    step = init_step
    logging.info(' ')
    logging.info('Start Training...')
    logging.info('init_step = %d' % init_step)
    logging.info('batch_size = %d' % args.batch_size)
    logging.info('negative_adversarial_sampling = %d' % args.negative_adversarial_sampling)
    logging.info('hidden_dim = %d' % args.hidden_dim)
    logging.info('gamma = %f' % args.gamma)
    logging.info('negative_adversarial_sampling = %s' % str(args.negative_adversarial_sampling))
    if args.negative_adversarial_sampling:
        logging.info('adversarial_temperature = %f' % args.adversarial_temperature)
    
    # Set valid dataloader as it would be evaluated during training
    
    if args.do_train:
        logging.info('learning_rate = %f' % current_learning_rate)
        logging.info('lr decay epoch = %s' % ('None' if args.lr_decay_epoch is None else str(args.lr_decay_epoch)))
        logging.info('lr decay rate = %f' % args.lr_decay_rate)
        logging.info('train txt = %s' % args.train_txt)
        logging.info(' ')

        training_logs = []
        args.lr_decay_epoch = [int(i) for i in args.lr_decay_epoch.split(',')]

        for step in range(init_step, args.max_steps):
            
            log = kge_model.train_step(kge_model, optimizer, train_iterator, args, step, viable_neg=viable_neg)
            training_logs.append(log)

            if step in args.lr_decay_epoch:
                current_learning_rate = current_learning_rate * args.lr_decay_rate
                logging.info('Change adam learning_rate to %f at step %d' % (current_learning_rate, step))

                optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, kge_model.parameters()),
                    lr=current_learning_rate
                )

            if (step + 1) % args.save_checkpoint_steps == 0:
                save_variable_list = {
                    'step': step,
                    'current_learning_rate': current_learning_rate,
                }
                save_model(kge_model, optimizer, save_variable_list, args, step)

            if step % args.log_steps == 0:
                metrics = {}
                for metric in training_logs[0].keys():
                    metrics[metric] = sum([log[metric] for log in training_logs])/len(training_logs)
                log_metrics('Training average', step, metrics)
                training_logs = []

            # evaluation
            if args.do_valid and (step + 1) % args.valid_steps == 0:
                logging.info('Evaluating on Valid Dataset...')
                metrics = kge_model.test_step(kge_model, valid_triples, class_test_triples_list_, lca_test_triples, all_true_triples, args,
                                                relation_category=args.do_test_relation_category, degree=degree)
                log_metrics('Valid', step, metrics)
                

            if args.evaluate_train and (step + 0) % args.valid_steps == 0:
                logging.info('Evaluating on Train Dataset...')
                metrics = kge_model.test_step(kge_model, train_triples, class_test_triples_list_, lca_test_triples, all_true_triples, args,
                                              relation_category=args.do_test_relation_category, degree=degree)
                log_metrics('Train', step, metrics)
                
        save_variable_list = {
            'step': step, 
            'current_learning_rate': current_learning_rate,
        }
        save_model(kge_model, optimizer, save_variable_list, args, step)
        
    if args.do_valid:
        logging.info('Evaluating on Valid Dataset...')
        metrics = kge_model.test_step(kge_model, valid_triples, class_test_triples_list_, lca_test_triples, all_true_triples, args,
                                      relation_category=args.do_test_relation_category, degree=degree)
        log_metrics('Valid', step, metrics)
    
    if args.do_test or args.do_test_relation_category:
        logging.info('Evaluating on Test Dataset...')
        metrics = kge_model.test_step(kge_model, test_triples, class_test_triples_list_, lca_test_triples, all_true_triples, args,
                                      relation_category=args.do_test_relation_category, degree=degree)
        log_metrics('Test', step, metrics)
    
    if args.evaluate_train:
        logging.info('Evaluating on Training Dataset...')
        metrics = kge_model.test_step(kge_model, train_triples, class_test_triples_list_, lca_test_triples, all_true_triples, args, relation_category=args.do_test_relation_category)
        log_metrics('Test', step, metrics)
        
if __name__ == '__main__':
    main(parse_args())
