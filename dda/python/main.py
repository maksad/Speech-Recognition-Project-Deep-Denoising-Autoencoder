import tensorflow as tf
import numpy as np
import pdb

from model import REG
from preprocessing import Synth, GenMatrix
from os.path import join
from utils import search_wav, get_best_data
from sklearn.model_selection import train_test_split

np.random.seed(1234567)

best_noise_data = [
    '1-4211-A-12.wav',
    '1-7456-A-13.wav',
    '1-9841-A-13.wav',
    '1-17367-A-10.wav',
    '1-18527-A-44.wav',
    '1-18527-B-44.wav',
    '1-19840-A-36.wav',
    '1-19872-B-36.wav',
    '1-21189-A-10.wav',
    '1-21896-A-35.wav',
    '1-23996-B-35.wav',
    '1-24074-A-43.wav',
    '1-27166-A-35.wav',
    '1-29532-A-16.wav',
    '1-32373-B-35.wav',
    '1-47709-A-16.wav',
    '3-181278-A-22.wav',
    '5-221567-A-22.wav',
    '5-262957-A-22.wav',
    '1-11687-A-47.wav',
]
best_training_data = [
    '/fin_words/S03/',
    '/fin_words/S06/',
    '/fin_words/S11/',
    '/fin_words/S15/',
    '/fin_words/S20/',
    '/fin_words/S22/',
    '/fin_words/S26/',
    '/fin_words/S27/',
    '/fin_words/S38/',
]


def synthesize_noisy_data(clean_dir, noise_dir, noisy_dir, feat, ncores):
    # ===========================================================
    # ===========       Synthesize Noisy Data        ============
    # ===========================================================
    clean_file_list = get_best_data(search_wav(clean_dir), best_training_data)
    noise_file_list = get_best_data(search_wav(noise_dir), best_noise_data)

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
    clean_dir = '../data_fin_words/fin_words'
    noise_dir = '../data/raw/noise' # use the normal noise
    noisy_dir =  '../fin_data/noisy'
    enhanced_dir =  '../fin_data/enhanced'
    tb_dir = '../fin_model/tb_logs'
    saver_dir = '../fin_model/saver'
    TRAIN = True
    TEST = True
    ncores = 20
    epochs = 20
    batch_size = 32
    lr = 1e-3
    feat = 'spec'; # spec/ mel/ mfcc
    train_task = 'same_noise'  # set task name for noting your dataset

    synthesize_noisy_data(clean_dir, noise_dir, noisy_dir, feat, ncores)

    create_training_matrix(noisy_dir, feat)

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
