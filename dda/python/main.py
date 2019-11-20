import tensorflow as tf
import numpy as np
import pdb


from model import REG
from preprocessing import Synth, GenMatrix
from os.path import join
from utils import search_wav
from sklearn.model_selection import train_test_split

np.random.seed(1234567)

def main():
    clean_dir = '../data/raw/clean'
    noise_dir = '../data/raw/noise'
    noisy_dir =  '../data/noisy'
    enhanced_dir =  '../data/enhanced'
    tb_dir = '../model/tb_logs'
    saver_dir = '../model/saver'
    TRAIN = True
    TEST = True
    ncores = 20
    epochs = 100
    batch_size = 32
    lr = 1e-3

    train_task = 'same_noise'  # set task name for noting your dataset

    # ===========================================================
    # ===========       Synthesize Noisy Data        ============
    # ===========================================================
    clean_file_list = search_wav(clean_dir)
    clean_train_list, clean_test_list = train_test_split(
        clean_file_list, test_size=0.2)
    noise_file_list = search_wav(noise_dir)[0:20]
    noise_train_list, noise_test_list = train_test_split(
        noise_file_list, test_size=0.2)
    noise_test_list = noise_train_list # test on the same noise

    print('--- Synthesize Training Noisy Data ---')
    train_noisy_dir = join(noisy_dir, 'train')
    sr_clean = 16000
    sr_noise = 44100
    snr_list = ['20dB', '10dB', '0dB']
    data_num = None  # set data_num to make training data numbers for different snr
    syn_train = Synth(clean_train_list, noise_train_list, sr_clean, sr_noise)
    syn_train.gen_noisy(snr_list, train_noisy_dir,
                        data_num=data_num, ADD_CLEAN=True, cpu_cores=ncores)
    print('--- Synthesize Testing Noisy Data ---')
    test_noisy_dir = join(noisy_dir, 'test')
    sr_clean = 16000
    sr_noise = 44100
    data_num = None # set data_num to make testing data numbers for different snr
    snr_list = ['15dB']
    syn_test = Synth(clean_test_list, noise_test_list, sr_clean, sr_noise)
    syn_test.gen_noisy(snr_list, test_noisy_dir,
                        data_num=data_num, ADD_CLEAN=True, cpu_cores=ncores)
    # ===========================================================
    # ===========       Create Training Matrix       ============
    # ===========================================================
    print('--- Generate Training Matrix ---')
    train_task = 'same_noise'  # set task name for noting your dataset
    training_files_dir = '../data/training_files'
    train_noisy_dir = join(noisy_dir, 'train')
    DEL_TRAIN_WAV = True
    gen_mat = GenMatrix(training_files_dir, train_task, train_noisy_dir)
    split_num = 50  # number of spliting files
    iter_num = 2  # set iter number to use multi-processing, cpu_cores = split_num/iter_num
    input_sequence = False  # set input data is sequence or not
    gen_mat.create_h5(split_num=split_num, iter_num=iter_num,
                      input_sequence=input_sequence,
                      DEL_TRAIN_WAV=DEL_TRAIN_WAV)

    # ===========================================================
    # ===========             Main Model             ============
    # ===========================================================
    print('--- Build Model ---')
    note = 'DDAE'
    date = '0720'
    split_num = 50
    training_files_dir = '../data/training_files'
    model = REG(tb_dir, saver_dir, train_task, date, gpu_num='3', note=note)
    model.build(init_learning_rate=1e-3, reuse=False)

    print('--- Train Model ---')
    model.train(training_files_dir, split_num, epochs, batch_size)

    print('--- Test Model ---')
    testing_data_dir = join(noisy_dir, 'test')
    result_dir = '../data/enhanced/{}_{}/'.format(note, date)
    num_test = 30 # Set this number to decide how many testing data you wanna use. (None => All)
    cpu_cores = 30
    test_saver = '{}_{}/{}/best_saver_{}'.format(
        saver_dir, note, date, train_task)
    model.test(testing_data_dir, result_dir,
               test_saver, cpu_cores, num_test)


if __name__ == '__main__':
    main()
