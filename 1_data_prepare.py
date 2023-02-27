import time
import yaml
import os
import torch
from utils.IO_func import read_file_list, array_to_binary_file, save_phone_label, save_word_label
from shutil import copyfile
from utils.transforms import Transform_Compose
from utils.transforms import FixMissingValues
from scipy.io import wavfile
import re, sys

def read_file_list(file_name):

    file_lists = []
    fid = open(file_name)
    for line in fid.readlines():
        line = line.strip()
        if len(line) < 1:
            continue
        file_lists.append(line)
    fid.close()

    return file_lists

def clean(s):
    s = str(s, "utf-8")
    return s.rstrip('\n').strip()
    
def mngu0_ema_loadin(ema_path, sel_channels):
    ## This piece of code was adopted from: https://github.com/karkirowle/articulatory_manipulation/blob/master/data_utils.py with minor changes.
    n_columns = 87
    columns = {}
    columns["time"] = 0
    columns["present"] = 1
    with open(ema_path, 'rb') as f:
        dummy_line = f.readline()  # EST File Track
        datatype = clean(f.readline()).split()[1]
        nframes = int(clean(f.readline()).split()[1])
        f.readline()  # Byte Order
        nchannels = int(clean(f.readline()).split()[1])

        while not 'CommentChar' in str(f.readline(), "utf-8"):
            pass
        f.readline()  # empty line
        line = clean(f.readline())
    
        while not "EST_Header_End" in line:
            channel_number = int(line.split()[0].split('_')[1]) + 2
            channel_name = line.split()[1]
            columns[channel_name] = channel_number
            line = clean(f.readline())
            
        ema_buffer = f.read()
        data = np.frombuffer(ema_buffer, dtype='float32')
        data_ = np.reshape(data, (-1, len(columns)))
    articulator_idx = [columns[articulator] for articulator in sel_channels]
    data_out = data_[:, articulator_idx]
    data_out = data_out*100  #initial data in  10^-5m , we turn it to mm
    if np.isnan(data_out).sum() != 0:
        # Build a cubic spline out of non-NaN values.
        spline = scipy.interpolate.splrep(np.argwhere(~np.isnan(data_out).ravel()), data_out[~np.isnan(data_out)], k=3)
        # Interpolate missing values and replace them.
        for j in np.argwhere(np.isnan(data_out)).ravel():
            data_out[j] = scipy.interpolate.splev(j, spline)
                
    return data_out
    
def mngu0_utt_loadin(utt_path):
    with open(utt_path, 'r') as f:
        lab = f.readlines()[4].split('\\')[1][1:]        
    return lab

def data_processing(args):

    config_path = args.conf_dir       
    config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)

    out_folder = os.path.join(args.buff_dir, 'data')
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    
    transforms = [FixMissingValues()] # default transforms
    data_path = config['corpus']['path']

    In_EMA_folder = os.path.join(data_path, config['corpus']['EMA_path'])
    In_WAV_folder = os.path.join(data_path, config['corpus']['WAV_path'])
    In_TXT_folder = os.path.join(data_path, config['corpus']['TXT_path'])
    
    filesets_path = os.path.join(config['corpus']['Filesets_path'], 'file_id_list.scp')
    file_id_list = read_file_list(filesets_path)
    sel_channels = config['corpus']['sel_channels']  
    
    In_EMA_list = glob.glob(In_EMA_folder + '/*.ema')
    In_WAV_list = glob.glob(In_WAV_folder + '/*.wav')
    In_TXT_list = glob.glob(In_TXT_folder + '/*.utt')
    
    for file_id in file_id_list:
        EMA_path = os.path.join(In_EMA_folder, file_id + '.ema')
        WAV_path = os.path.join(In_WAV_folder, file_id + '.wav')
        TXT_path = os.path.join(In_TXT_folder, file_id + '.utt')    
        
        ema_data = mngu0_ema_loadin(In_EMA_list[i], sel_channels)
        wrd = mngu0_utt_loadin(In_TXT_list[i])
        out_line = file_id + '\t' + wrd + '\n'    
     
    

            
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_dir', default = 'conf/SSR_conf.yaml')
    parser.add_argument('--buff_dir', default = 'current_exp')

    args = parser.parse_args()
    data_processing(args)
