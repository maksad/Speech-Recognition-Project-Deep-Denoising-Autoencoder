import tensorflow as tf
import numpy as np
import pdb

from model import REG
from preprocessing import Synth, GenMatrix
from os.path import join
from utils import search_wav, get_best_data
from sklearn.model_selection import train_test_split

import argparse

parser = argparse.ArgumentParser()
paser.add_argument('--training_noisy_dir', type=str, default='./data/train_data/noisy')
paser.add_argument('--training_clean_dir', type=str, default='./data/train_data/clean')
paser.add_argument('--enhanced_dir', type=str, default= './data/enhanced')
paser.add_argument('--model_dir', type=str, default='./model')
paser.add_argument('--feat', type=str, default='spec', help='spec/ mel/ mfcc')
paser.add_argument('--train_task', type=str, default='same_noise', help='set task name for noting your dataset')
paser.add_argument('--train', type=str, default=True)
paser.add_argument('--ncores', type=str, default=20)
paser.add_argument('--epochs', type=str, default=20)
paser.add_argument('--batch_size', type=str, default=32)
paser.add_argument('--lr', type=str, default=1e-3)
args = parser.parse_args()


np.random.seed(1234567)


def synthesize_noisy_data():
    # ===========================================================
    # ===========       Synthesize Noisy Data        ============
    # ===========================================================
    clean_file_list = search_wav(clean_dir)
    noise_file_list = search_wav(noise_dir)

    clean_train_list, clean_test_list = train_test_split(clean_file_list, test_size=0.2)
    noise_train_list, noise_test_list = train_test_split(noise_file_list, test_size=0.2)
    noise_test_list = noise_train_list # test on the same noise

    print('--- Synthesize Training Noisy Data ---')
    train_noisy_dir = join(noisy_dir, 'train')
    sr_clean = 16000
    sr_noise = 44100
    snr_list = ['20dB', '10dB', '0dB']
    data_num = None  # set data_num to make training data numbers for different snr
    syn_train = Synth(clean_train_list, noise_train_list, feat, sr_clean, sr_noise)
    syn_train.gen_noisy(
        snr_list, train_noisy_dir, data_num=data_num, ADD_CLEAN=True, cpu_cores=ncores
    )
    print('--- Synthesize Testing Noisy Data ---')
    test_noisy_dir = join(noisy_dir, 'test')
    sr_clean = 16000
    sr_noise = 44100
    data_num = None # set data_num to make testing data numbers for different snr
    snr_list = ['15dB']
    syn_test = Synth(clean_test_list, noise_test_list, feat,  sr_clean, sr_noise)
    syn_test.gen_noisy(
        snr_list, test_noisy_dir, data_num=data_num, ADD_CLEAN=True, cpu_cores=ncores
    )


def create_training_matrix(noisy_dir, feat):
    # ===========================================================
    # ===========       Create Training Matrix       ============
    # ===========================================================
    print('--- Generate Training Matrix ---')
    train_task = 'same_noise'  # set task name for noting your dataset
    training_files_dir = '../fin_data/training_files';
    train_noisy_dir = join(noisy_dir, 'train');
    DEL_TRAIN_WAV = True;
    gen_mat = GenMatrix(training_files_dir, train_task, train_noisy_dir);
    split_num = 50  # number of spliting files
    iter_num = 2  # set iter number to use multi-processing, cpu_cores = split_num/iter_num
    input_sequence = False  # set input data is sequence or not
    gen_mat.create_h5(
        split_num=split_num,
        iter_num=iter_num,
        feat=feat,
        input_sequence=input_sequence,
        DEL_TRAIN_WAV=DEL_TRAIN_WAV
    )


def main():
    synthesize_noisy_data()

    create_training_matrix()

    # ===========================================================
    # ===========             Main Model             ============
    # ===========================================================
    print('--- Build Model ---')
    note = 'DDAE'
    date = '0720'
    split_num = 50
    training_files_dir = '../fin_data/training_files'
    model = REG(tb_dir, saver_dir, train_task, date, gpu_num='3', note=note)
    model.build(init_learning_rate=1e-3, reuse=False, feat=feat)

    print('--- Train Model ---')
    model.train(training_files_dir, split_num, epochs, batch_size)

    print('--- Test Model ---')
    testing_data_dir = join(noisy_dir, 'test')
    result_dir = '../fin_data/enhanced/{}_{}/'.format(note, date)
    num_test = 30 # Set this number to decide how many testing data you wanna use. (None => All)
    cpu_cores = 30
    test_saver = '{}_{}/{}/best_saver_{}'.format(saver_dir, note, date, train_task)
    model.test(testing_data_dir, result_dir, feat, test_saver, cpu_cores, num_test)


if __name__ == '__main__':
    main()
