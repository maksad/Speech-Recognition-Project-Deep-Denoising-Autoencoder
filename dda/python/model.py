#import tensorflow as tf
import tensorflow.compat.v1 as tf
import h5py
import numpy as np
import scipy

import os
from os.path import join
from tqdm import tqdm
import matplotlib.pyplot as plt
tqdm.monitor_interval = 0
from utils import (
    np_REG_batch,
    search_wav,
    wav2spec,
    wav2melspec,
    wav2mfcc,
    melspec2wav,
    spec2wav,
    mfccupload,
    copy_file
)
#tf.disable_v2_behavior()

class REG:
    def __init__(self, model_dir, gpu_num, h5_file_name):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_num

        self.model_dir = model_dir
        self.h5_file_name = h5_file_name
        self.config = tf.ConfigProto(allow_soft_placement=True)
        self.config.gpu_options.allow_growth = True

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

    def build(self, init_learning_rate, reuse, feat):
        self.init_learning_rate = init_learning_rate
        self.name = 'REG_Net_{}'.format(feat)
        tf.disable_eager_execution()

        # regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)

        if feat=='spec':
            input_no = 257;
        elif feat=='mel':
            input_no=128;
        elif feat=='mfcc':
            input_no=26

        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()

            with tf.variable_scope('Intputs'):
                self.x_noisy = tf.placeholder(tf.float32, shape=[None, input_no], name='x')
            with tf.variable_scope('Outputs'):
                self.y_clean = tf.placeholder(tf.float32, shape=[None, input_no], name='y_clean')
            with tf.name_scope('weights'):
                w = {
                    'w_o': tf.get_variable(
                        'WO', shape=[512, input_no],
                        # regularizer=regularizer,
                        initializer=tf.glorot_normal_initializer(seed=None, dtype=tf.float32)
                    ),
                    'w_1': tf.get_variable(
                        'W1', shape=[input_no, 512],
                        # regularizer=regularizer,
                        initializer=tf.glorot_normal_initializer(seed=None, dtype=tf.float32)
                    ),
                    'w_2': tf.get_variable(
                        'W2',
                        shape=[512, 512],
                        # regularizer=regularizer,
                        initializer=tf.glorot_normal_initializer(seed=None, dtype=tf.float32)
                    ),
                    'w_3': tf.get_variable(
                        'W3',
                        shape=[512, 512],
                        # regularizer=regularizer,
                        initializer=tf.glorot_normal_initializer(seed=None, dtype=tf.float32)
                    ),
                    'w_4': tf.get_variable(
                        'W4',
                        shape=[512, 512],
                        # regularizer=regularizer,
                        initializer=tf.glorot_normal_initializer(seed=None, dtype=tf.float32)
                    )
                }

            with tf.name_scope('bias'):
                b = {
                    'b_o': tf.get_variable(
                        "bO",
                        shape=[1, input_no],
                        initializer=tf.constant_initializer(value=0, dtype=tf.float32)
                    ),
                    'b_1': tf.get_variable(
                        "b1",
                        shape=[1, 512],
                        initializer=tf.constant_initializer(value=0, dtype=tf.float32)
                    ),
                    'b_2': tf.get_variable(
                        "b2",
                        shape=[1, 512],
                        initializer=tf.constant_initializer(value=0, dtype=tf.float32)
                    ),
                    'b_3': tf.get_variable(
                        "b3",
                        shape=[1, 512],
                        initializer=tf.constant_initializer(value=0, dtype=tf.float32)
                    ),
                    'b_4': tf.get_variable(
                        "b4",
                        shape=[1, 512],
                        initializer=tf.constant_initializer(value=0, dtype=tf.float32)
                    )
                }

            with tf.variable_scope('DNN'):
                layer_1 = tf.nn.leaky_relu(tf.add(tf.matmul(self.x_noisy, w['w_1']), b['b_1']))
                layer_2 = tf.nn.leaky_relu(tf.add(tf.matmul(layer_1, w['w_2']), b['b_2']))
                layer_3 = tf.nn.leaky_relu(tf.add(tf.matmul(layer_2, w['w_3']), b['b_3']))
                layer_4 = tf.nn.leaky_relu(tf.add(tf.matmul(layer_3, w['w_4']), b['b_4']))
                self.reg_layer = tf.add(tf.matmul(layer_4, w['w_o']), b['b_o'])


            with tf.name_scope('reg_loss'):
                self.loss_reg = tf.losses.mean_squared_error(
                    self.y_clean, self.reg_layer)

                tf.summary.scalar('Loss reg', self.loss_reg)

            with tf.name_scope("exp_learning_rate"):
                self.global_step = tf.Variable(0, trainable=False)
                self.exp_learning_rate = tf.train.exponential_decay(
                    self.init_learning_rate,
                    global_step=self.global_step,
                    decay_steps=500000,
                    decay_rate=0.95,
                    staircase=False
                )
                tf.summary.scalar('Learning rate', self.exp_learning_rate)

            optimizer = tf.train.AdamOptimizer(self.init_learning_rate)
            gradients, v = zip(*optimizer.compute_gradients(self.loss_reg))
            gradients, _ = tf.clip_by_global_norm(gradients, 0.5)
            self.optimizer = optimizer.apply_gradients(
                zip(gradients, v),
                global_step=self.global_step
            )
            self.saver = tf.train.Saver()


    def train(self, h5_dir_name, split_num, epochs, batch_size, base_model_dir=None):
        if (
            not (
                base_model_dir and
                os.path.normpath(self.model_dir) == os.path.normpath(
                    '/'.join(base_model_dir.split('/')[:-1])
                )
            )
            and tf.gfile.Exists(self.model_dir)
        ):
            tf.gfile.DeleteRecursively(self.model_dir)
            tf.gfile.MkDir(self.model_dir)
        best_reg_loss = 700.

        with tf.Session(config=self.config) as sess:
            tf.global_variables_initializer().run()
            writer = tf.summary.FileWriter(self.model_dir, sess.graph,  max_queue=10)
            merge_op = tf.summary.merge_all()

            if base_model_dir:
                print('Loading base model: {}'.format(base_model_dir))
                self.saver.restore(sess=sess, save_path=base_model_dir)
                sess.run(tf.local_variables_initializer())

            print('Start Training')
            # set early stopping
            patience = 10
            FLAG = False
            min_delta = 0.01
            step = 0
            epochs = range(epochs)
            loss_list = []

            for epoch in tqdm(epochs):
                shuffle_list = np.arange(split_num)
                np.random.shuffle(shuffle_list)
                loss_reg_tmp = 0.
                count = 0

                for i in tqdm(shuffle_list):
                    data_name = join(h5_dir_name, '{}_{}.h5'.format(self.h5_file_name, i))
                    data_file = h5py.File(data_name, 'r')
                    # import IPython; IPython.embed()
                    clean_data = data_file['clean_data']
                    noisy_data = data_file['noisy_data']
                    data_len = len(clean_data)
                    data_batch = np_REG_batch(noisy_data, clean_data, batch_size, data_len)

                    for batch in range(int(data_len / batch_size)):
                        noisy_batch, clean_batch = next(data_batch), next(data_batch)
                        feed_dict = {
                            self.x_noisy: noisy_batch,
                            self.y_clean: clean_batch
                        }
                        _, loss_var1, summary_str = sess.run(
                            [self.optimizer, self.loss_reg, merge_op],
                            feed_dict=feed_dict
                        )
                        loss_reg_tmp += loss_var1
                        count += 1
                        writer.add_summary(summary_str, step)
                        step += 1

                if epoch % 1 == 0:
                    loss_reg_tmp /= count
                    loss_var = loss_reg_tmp

                    print('[epoch {}] Loss Reg:{}'.format(
                        int(epoch), loss_reg_tmp))

                    loss_list.append(loss_var)
                    if loss_var <= (best_reg_loss - min_delta):
                        best_reg_loss = loss_var
                        self.saver.save(sess=sess, save_path=join(self.model_dir, 'trained_model'))
                        patience = 10
                        print('Best Reg Loss: ', best_reg_loss)
                    else:
                        print('Not improve Loss:', best_reg_loss)
                        if FLAG == True:
                            patience -= 1


                    plt.figure()
                    plt.plot(loss_list, label='loss')
                    plt.xlabel('Epochs')
                    plt.ylabel('MSE loss')
                    plot_name = join(self.model_dir, 'loss_history.png')
                    plt.legend()
                    plt.savefig(plot_name)

                if patience == 0 and FLAG == True:
                    print('Early Stopping ! ! !')
                    break

    def test(self, testing_data_dir, enhanced_dir, feat, model_dir, num_test=False):
        print('Start Testing')
        tmp_list = search_wav(testing_data_dir)

        if num_test:
            test_list = np.random.choice(tmp_list, num_test)
        else:
            test_list = tmp_list

        print('All testing data number:', len(test_list))

        if not os.path.exists(enhanced_dir):
            os.makedirs(enhanced_dir)

        with tf.Session(config=self.config) as sess:
            self.saver.restore(sess=sess, save_path=model_dir)

            for file in tqdm(test_list):
                hop_length = 256
                noise_name, subdir_name, file_name = file.split('/')[-3:]
                try:
                    # noise_name, subdir_name, file_name = file.split('/')[-3:]
                    # clean_file = join(testing_data_dir, '/'.join([subdir_name, file_name]))
                    noisy_file = file
                except:
                    raise NotImplementedError('File name was not found')
                    # snr, noise_name, clean_name = file.split('/')[-1].split('_')

                new_file_dir = '/'.join([noise_name, subdir_name])
                enhanced_file_dir = join(enhanced_dir, new_file_dir)
                if not os.path.exists(enhanced_file_dir):
                    os.makedirs(enhanced_file_dir)

                REG_file = join(enhanced_file_dir, file_name)

                if feat=='spec':
                    X_in_seq = wav2spec(
                        noisy_file, sr=16000, forward_backward=False, SEQUENCE=False, norm=False, hop_length=hop_length
                    )
                elif feat=='mel':
                    X_in_seq = wav2melspec(
                        noisy_file, sr=16000, forward_backward=False, SEQUENCE=False, norm=False, hop_length=hop_length
                    )
                elif feat=='mfcc':
                    X_in_seq = wav2mfcc(
                        noisy_file, sr=16000, forward_backward=False, SEQUENCE=False, norm=False, hop_length=hop_length
                    )
                re_reg = sess.run([self.reg_layer], feed_dict={self.x_noisy: X_in_seq})[:][0]
                if feat=='spec':
                    spec2wav(noisy_file, 16000, REG_file, re_reg, hop_length=hop_length);
                if feat=='mel':
                    melspec2wav(noisy_file, 16000, REG_file, re_reg, hop_length=hop_length);
                if feat=='mfcc':
                    mfccupload(noisy_file, 16000, REG_file, re_reg, hop_length=hop_length);
