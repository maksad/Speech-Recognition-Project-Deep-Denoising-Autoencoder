import os
from os.path import join
import shutil
from sklearn.model_selection import train_test_split

import tensorflow as tf
import numpy as np
import pdb
import argparse

from model import REG
from preprocessing import Synth, GenMatrix
from utils import search_wav, str2bool


parser = argparse.ArgumentParser()
parser.add_argument('--training_data_dir', type=str,      default='./data/train_data')
parser.add_argument('--testing_data_dir',  type=str,      default='./data/test_data/noisy')
parser.add_argument('--enhanced_dir',      type=str,      default= './data/enhanced')
parser.add_argument('--h5_dir_name',       type=str,      default='./data/train_data/training_files_h5')
parser.add_argument('--h5_file_name',      type=str,      default='file', help='set task name for noting your dataset')
parser.add_argument('--model_dir',         type=str,      default='./trained_model')
parser.add_argument('--base_model_dir',    type=str,      default=None)
parser.add_argument('--feat',              type=str,      default='spec', help='spec, mel, mfcc')
parser.add_argument('--num_cpu',           type=int,      default=5)
parser.add_argument('--epochs',            type=int,      default=50)
parser.add_argument('--batch_size',        type=int,      default=32)
parser.add_argument('--split_number',      type=int,      default=100)
parser.add_argument('--lr',                type=float,    default=1e-3)
parser.add_argument('--train',             type=str2bool, default=True)
parser.add_argument('--test',              type=str2bool, default=True)
parser.add_argument('--keep_training_matrix',type=str2bool, default=False)

args = parser.parse_args()


np.random.seed(1234567)


def create_training_matrix():
    print('--- Generate Training Matrix ---')
    gen_mat = GenMatrix(
        args.h5_dir_name,
        args.h5_file_name,
        join(args.training_data_dir, 'noisy'),
        join(args.training_data_dir, 'clean'),
    )
    gen_mat.create_h5(
        split_num=args.split_number, # number of spliting files
        iter_num=args.num_cpu,  # set iter number to use multi-processing, cpu_cores = split_num/iter_num
        feat=args.feat,
        input_sequence=False, # set input data is sequence or not
    )


def main():
    if args.train:
        if os.path.exists(args.h5_dir_name) and not args.keep_training_matrix:
            shutil.rmtree(args.h5_dir_name)
        if not args.keep_training_matrix:
            create_training_matrix()
        if not os.path.exists(args.model_dir):
            os.makedirs(args.model_dir)

    print('--- Build Model ---')
    model = REG(args.model_dir, gpu_num='3', h5_file_name=args.h5_file_name)
    model.build(init_learning_rate=1e-3, reuse=False, feat=args.feat)

    if args.train:
        print('--- Train Model ---')
        base_model_dir = join(args.base_model_dir, 'trained_model') if args.base_model_dir else None

        model.train(
            args.h5_dir_name,
            args.split_number,
            args.epochs,
            args.batch_size,
            base_model_dir=base_model_dir,
        )

    if args.test:
        print('--- Test Model ---')
        model.test(
            args.testing_data_dir,
            args.enhanced_dir,
            args.feat,
            join(args.model_dir, 'trained_model'),
            num_test=30, # set this number to decide how many testing data you wanna use. (None => All)
        )


if __name__ == '__main__':
    main()
