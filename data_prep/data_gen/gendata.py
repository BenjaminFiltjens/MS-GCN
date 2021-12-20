import os

import argparse
from tqdm import tqdm
import numpy as np
from data_prep.data_gen.read_skeleton import read_xyz

max_body = 1
num_joint = 9


def gendata(data_path,
            out_path):
    file_list = []
    for path in os.listdir(data_path):
        file_list.append(path)

    for filename in tqdm(file_list):
        if filename.endswith(('_input.csv')):
            data = read_xyz(data_path+filename, max_body=max_body, num_joint=num_joint)
            np.save('{}/{}.npy'.format(out_path, str(filename[:-4])), data)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='FOG Data Converter.')
    parser.add_argument(
        '--data_path', default='D:\\dataset_output\\')
    parser.add_argument('--out_folder',
                        default='D:\\dataset_npy\\')

    arg = parser.parse_args()

    out_path = arg.out_folder
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    gendata(
        arg.data_path,
        out_path)
