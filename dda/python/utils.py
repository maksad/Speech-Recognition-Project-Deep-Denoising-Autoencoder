
import librosa
import numpy as np
import scipy
import os
import h5py
from glob import iglob
from shutil import copy2
from os.path import join
import os


epsilon = np.finfo(float).eps


def np_REG_batch(data1, data2, batch_size, data_len):
    n_start = 0
    n_end = batch_size
    l = data_len
    while True:
        if n_end >= l:
            yield data1[n_start:]
            yield data2[n_start:]
            n_start = 0
            n_end = batch_size
        else:
            yield data1[n_start:n_end]
            yield data2[n_start:n_end]

        n_start = n_end
        n_end += batch_size


def search_wav(data_path):
    file_list = []
    for filename in iglob('{}/**/*.WAV'.format(data_path), recursive=True):
        file_list.append(str(filename))
    for filename in iglob('{}/**/*.wav'.format(data_path), recursive=True):
        file_list.append(str(filename))
    return file_list

def split_list(alist, wanted_parts=20):
    length = len(alist)
    return [alist[i * length // wanted_parts: (i + 1) * length // wanted_parts]
            for i in range(wanted_parts)]

def wav2spec(wavfile, sr, forward_backward=None, SEQUENCE=None, hop_length=256, norm=False):
    # Note:This function return three different kind of spec for training and
    # testing
    y, sr = librosa.load(wavfile, sr, mono=True)
    NUM_FFT = 512

    D = librosa.stft(y,
                     n_fft=NUM_FFT,
                     hop_length=hop_length,
                     win_length=512,
                     window=scipy.signal.hann)
  #  librosa.core.stft(y, n_fft=2048, hop_length=None, win_length=None, window='hann', center=True, dtype=<class 'numpy.complex64'>, pad_mode='reflect')
    D = D + epsilon
    Sxx = np.log10(abs(D)**2)
    Sxx_r = np.array(Sxx);
    Sxx_r = np.array(Sxx_r).T
    return Sxx_r

def wav2melspec(wavfile, sr, forward_backward=False, SEQUENCE=None, norm=False, hop_length=256):
    # Note:This function return three different kind of spec for training and
    # testing
    y, sr = librosa.load(wavfile, sr, mono=True)
    NUM_FFT = 512

    D = librosa.feature.melspectrogram(y,
                     n_fft=NUM_FFT,
                     hop_length=hop_length,
                     win_length=512,
                     window=scipy.signal.hann)

    D = D + epsilon
    Sxx = np.log10(abs(D)**2)
    Sxx_r = np.array(Sxx);
    Sxx_r = np.array(Sxx_r).T
    return Sxx_r

def wav2mfcc(wavfile, sr, forward_backward=False, SEQUENCE=None, norm=False, hop_length=256):
    # Note:This function return three different kind of spec for training and
    # testing
    y, sr = librosa.load(wavfile, sr, mono=True)
    mfcc = librosa.feature.mfcc(y, win_length=512, n_fft=512,
         hop_length=256,
         sr = sr, n_mfcc = 13)
    delta = librosa.feature.delta(mfcc);
    mfcc_new = np.concatenate([mfcc, delta]);
    return mfcc_new.T


def spec2wav(wavfile, sr, output_filename, spec_test, hop_length=256):

    y, sr = librosa.load(wavfile, sr, mono=True)
    D = librosa.stft(y,
                     n_fft=512,
                     hop_length=hop_length,
                     win_length=512,
                     window=scipy.signal.hann)
    D = D + epsilon
    phase = np.exp(1j * np.angle(D))
    Sxx_r_tmp = np.array(spec_test)
    Sxx_r_tmp = np.sqrt(10**Sxx_r_tmp)
    Sxx_r = Sxx_r_tmp.T
    reverse = np.multiply(Sxx_r, phase)

    result = librosa.istft(reverse,
                           hop_length=hop_length,
                           win_length=512,
                           window=scipy.signal.hann)
    melspec = librosa.feature.melspectrogram(S =  Sxx_r, n_fft=512, sr=sr);
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(melspec), sr=sr, n_mfcc=13, n_fft=512, win_length=512);
    delta = librosa.feature.delta(mfcc);
    mfcc_new = np.concatenate([mfcc, delta]);
    if(os.path.exists(output_filename[0:-4] +'.mfc')):
        os.remove(output_filename[0:-4] +'.mfc')
    np.savetxt(output_filename[0:-4] +'.mfc', mfcc_new);

    y_out = librosa.util.fix_length(result, len(y), mode='edge')
    y_out = y_out/np.max(np.abs(y_out))
    if(np.max(abs(y_out)>1)):
        y_out = y_out/max(abs(y_out));
#    librosa.output.write_wav(
#        output_filename, (y_out * maxv).astype(np.int16), sr)
    librosa.output.write_wav(
        output_filename, y_out, sr)

def melspec2wav(wavfile, sr, output_filename, spec_test, hop_length=256):

    y, sr = librosa.load(wavfile, sr, mono=True)
#    D = librosa.feature.melspectrogram(y,
#                     n_fft=512,
#                     hop_length=hop_length,
#                     win_length=512,
#                     window=scipy.signal.hann)
  #  D = D.T + epsilon
  #  phase = np.exp(1j * np.angle(D))
    Sxx_r_tmp = np.array(spec_test)
    reverse = np.sqrt(10**Sxx_r_tmp)
    Sxx_r = reverse.T
   # reverse = np.multiply(Sxx_r_tmp, phase)

#    result = librosa.feature.inverse.mel_to_audio(reverse,
#                           hop_length=hop_length,
#                           win_length=512,
#                           window=scipy.signal.hann)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(Sxx_r), sr=sr, n_mfcc=13, n_fft=512, win_length = 512);
    delta = librosa.feature.delta(mfcc);
    mfcc_new = np.concatenate([mfcc, delta]);
    if(os.path.exists(output_filename[0:-4] +'.mfc')):
        os.remove(output_filename[0:-4] +'.mfc')
    np.savetxt(output_filename[0:-4] +'.mfc', mfcc_new);

    librosa.output.write_wav( output_filename, y, sr) #uploads noisy file

def mfccupload(wavfile, sr, output_filename, mfcc, hop_length=256): #not doing the actual converting; just uploading

    y, sr = librosa.load(wavfile, sr, mono=True)
    if(os.path.exists(output_filename[0:-4] +'.mfc')):
        os.remove(output_filename[0:-4] +'.mfc')
    np.savetxt(output_filename[0:-4] +'.mfc', mfcc.T);
    librosa.output.write_wav(
        output_filename, y, sr)


def copy_file(input_file, output_file, hop_length):
    copy2(input_file, output_file)
    y, sr = librosa.load(input_file, sr=16000, mono=True)
    mfcc = librosa.feature.mfcc(y,
                 hop_length=hop_length, n_fft = 512, win_length=512,
                 sr = sr, n_mfcc = 13)
    delta = librosa.feature.delta(mfcc);
    mfcc_new = np.concatenate([mfcc, delta]);
    if(os.path.exists(output_file[0:-4] +'.mfc')):
        os.remove(output_file[0:-4] +'.mfc')
    np.savetxt(output_file[0:-4] +'.mfc', mfcc_new);



def _gen_noisy(clean_file_list, noise_file_list, save_dir, snr, sr_clean, sr_noise, num=None):
    SNR = float(snr.split('dB')[0])
    clean_file = clean_file_list[num]
    noise_file = noise_file_list[num]
    clean_name = clean_file.split('/')[-1].split('.')[0]
    noise_name = noise_file.split('/')[-1].split('.')[0]
    y_clean, sr_clean = librosa.load(clean_file, sr_clean, mono=True)
    #### scipy cannot conver TIMIT format ####

    clean_pwr = sum(abs(y_clean)**2) / len(y_clean)
    y_noise, sr_noise = librosa.load(noise_file, sr_noise, mono=True)

    tmp_list = []
    if len(y_noise) < len(y_clean):
        tmp = (len(y_clean) // len(y_noise)) + 1
        y_noise = np.array([x for j in [y_noise] * tmp for x in j])
        y_noise = y_noise[:len(y_clean)]
    else:
        y_noise = y_noise[:len(y_clean)]
    y_noise = y_noise - np.mean(y_noise)
    noise_variance = clean_pwr / (10**(SNR / 10))
    noise = np.sqrt(noise_variance) * y_noise / np.std(y_noise)
    y_noisy = y_clean + noise
    maxv = np.iinfo(np.int16).max
    save_name = '{}_{}_{}.wav'.format(snr, noise_name, clean_name)
    if(np.max(abs(y_noisy)>1)):
        y_noisy = y_noisy/max(abs(y_noisy));
#    librosa.output.write_wav(
#        '/'.join([save_dir, save_name]), (y_noisy * maxv).astype(np.int16), sr_clean)
    librosa.output.write_wav(
        '/'.join([save_dir, save_name]), y_noisy, sr_clean)

def _gen_clean(clean_file_list, save_dir, snr, num=None):
    sr_clean = 16000
    noise_name = 'n0'
    clean_file = clean_file_list[num]
    y_clean, sr_clean = librosa.load(clean_file, sr_clean, mono=True)


    clean_name = clean_file.split('/')[-1].split('.')[0]
    if(np.max(abs(y_clean)>1)):
        y_clean = y_clean/max(abs(y_clean));
    save_name = '{}_{}_{}.wav'.format(snr, noise_name, clean_name)
#    librosa.output.write_wav(
#        '/'.join([save_dir, save_name]), (y_clean * maxv).astype(np.int16), sr_clean)
    librosa.output.write_wav(
        '/'.join([save_dir, save_name]), y_clean, sr_clean)

def _create_split_h5(clean_split_list,
                     noisy_split_list, feat,
                     save_dir,
                     file_name,
                     input_sequence=False,
                     split_num=None):
    noisy_tmp = []
    clean_tmp = []
    for clean_file, noisy_file in zip(clean_split_list[split_num], noisy_split_list[split_num]):
        # you can set noisy data is sequence or not
        if feat=='spec':
            noisy_spec = wav2spec(
                noisy_file, sr=16000, forward_backward=False, SEQUENCE=input_sequence, norm=False, hop_length=256)
            clean_spec = wav2spec(
                clean_file, sr=16000, forward_backward=False, SEQUENCE=input_sequence, norm=False, hop_length=256)
        elif feat=='mel':
            noisy_spec = wav2melspec(
                noisy_file, sr=16000, forward_backward=False, SEQUENCE=input_sequence, norm=False, hop_length=256)
            clean_spec = wav2melspec(
                clean_file, sr=16000, forward_backward=False, SEQUENCE=input_sequence, norm=False, hop_length=256)
        elif feat=='mfcc':
            noisy_spec = wav2mfcc(
                noisy_file, sr=16000, forward_backward=False, SEQUENCE=input_sequence, norm=False, hop_length=256)
            clean_spec = wav2mfcc(
                clean_file, sr=16000, forward_backward=False, SEQUENCE=input_sequence, norm=False, hop_length=256)

        noisy_tmp.append(noisy_spec)
        clean_tmp.append(clean_spec)
        # if count % int(len(clean_split_list[split_num]) / 10) == 0:
        #     tmp = int(len(clean_split_list[split_num]) / 10)
            # print('Part {} {}%'.format(split_num, 10 * count / tmp))
        # count += 1
        if clean_spec.shape[0] == noisy_spec.shape[0]:
            continue
        else:
            print('Mismatch', noisy_file, clean_file)
            print('Clean shape:', clean_spec.shape)
            print('Noisy shape: ', noisy_spec.shape)

    noisy_data = np.vstack(noisy_tmp)
    y_clean_data = np.vstack(clean_tmp)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with h5py.File(join(save_dir, '{}_{}.h5'.format(file_name, split_num)), 'w') as hf:
        hf.create_dataset('noisy_data', data=noisy_data)
        hf.create_dataset('clean_data', data=y_clean_data)

    del noisy_data
    del y_clean_data


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
