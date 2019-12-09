from os.path import join
from sklearn.model_selection import train_test_split

import tensorflow as tf
import numpy as np
import pdb
import argparse

from model import REG
from preprocessing import Synth, GenMatrix
from utils import search_wav, get_best_data


parser = argparse.ArgumentParser()
parser.add_argument('--training_noisy_dir', type=str, default='./data/train_data/noisy')
parser.add_argument('--training_clean_dir', type=str, default='./data/train_data/clean')
parser.add_argument('--enhanced_dir', type=str, default= './data/enhanced')
parser.add_argument('--save_h5_dir', type=str, default='./data/training_files_h5')
parser.add_argument('--save_h5_name', type=str, default='file', help='set task name for noting your dataset')
parser.add_argument('--model_dir', type=str, default='./model')
parser.add_argument('--feat', type=str, default='spec', help='spec, mel, mfcc')
parser.add_argument('--train', type=str, default=True)
parser.add_argument('--ncores', type=str, default=20)
parser.add_argument('--epochs', type=str, default=20)
parser.add_argument('--batch_size', type=str, default=32)
parser.add_argument('--lr', type=str, default=1e-3)
args = parser.parse_args()


np.random.seed(1234567)


def create_training_matrix():
    print('--- Generate Training Matrix ---')
    gen_mat = GenMatrix(
        args.save_h5_dir,
        args.save_h5_name,
        args.training_noisy_dir,
        args.training_clean_dir,
    )
    gen_mat.create_h5(
        split_num=100, # number of spliting files
        iter_num=6,  # set iter number to use multi-processing, cpu_cores = split_num/iter_num
        feat=args.feat,
        input_sequence=False, # set input data is sequence or not
    )


def main():
    create_training_matrix()

    # print('--- Build Model ---')
    # note = 'DDAE'
    # date = '0720'
    # split_num = 50
    # training_files_dir = '../fin_data/training_files'
    # model = REG(tb_dir, saver_dir, train_task, date, gpu_num='3', note=note)
    # model.build(init_learning_rate=1e-3, reuse=False, feat=feat)

    # print('--- Train Model ---')
    # model.train(training_files_dir, split_num, epochs, batch_size)

    # print('--- Test Model ---')
    # testing_data_dir = join(noisy_dir, 'test')
    # result_dir = '../fin_data/enhanced/{}_{}/'.format(note, date)
    # num_test = 30 # Set this number to decide how many testing data you wanna use. (None => All)
    # cpu_cores = 30
    # test_saver = '{}_{}/{}/best_saver_{}'.format(saver_dir, note, date, train_task)
    # model.test(testing_data_dir, result_dir, feat, test_saver, cpu_cores, num_test)


if __name__ == '__main__':
    main()
