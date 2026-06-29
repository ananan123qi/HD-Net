import os
import glob
import numpy as np
import scipy.io as sio
import h5py

def load_mat(file_path):
    try:

        data = sio.loadmat(file_path)

        data = {k: v for k, v in data.items() if not k.startswith('__')}
        return data
    except NotImplementedError:

        with h5py.File(file_path, 'r') as f:
            data = {}
            for k in f.keys():
                data[k] = np.array(f[k])


            return data

def convert_test_sets():
    base_dir = 'data/hdnet'


    f1 = os.path.join(base_dir, 'test_calbased.mat')
    if os.path.exists(f1):
        print(f"Converting {f1}...")
        data1 = load_mat(f1)
        np.savez(f1.replace('.mat', '.npz'), **data1)
        print("Done.")


    f2 = os.path.join(base_dir, 'test_calfree.mat')
    if os.path.exists(f2):
        print(f"Converting {f2}...")
        data2 = load_mat(f2)
        np.savez(f2.replace('.mat', '.npz'), **data2)
        print("Done.")

def merge_and_convert_train_chunks():
    chunks_dir = 'data/hdnet/train_chunks'
    if not os.path.exists(chunks_dir):
        print("Train chunks dir not found")
        return

    chunk_files = sorted(glob.glob(os.path.join(chunks_dir, 'chunk_*.mat')))
    print(f"Found {len(chunk_files)} train chunks.")

    merged_data = {}

    for i, cf in enumerate(chunk_files):
        print(f"Reading {cf} ({i+1}/{len(chunk_files)})...")
        data = load_mat(cf)

        for k, v in data.items():
            if k not in merged_data:
                merged_data[k] = []
            merged_data[k].append(v)

    print("Concatenating arrays...")
    for k in merged_data.keys():
        try:



            merged_data[k] = np.concatenate(merged_data[k], axis=0)
        except ValueError as e:
            print(f"Could not concatenate key {k}: {e}. Saving as object array.")
            merged_data[k] = np.array(merged_data[k], dtype=object)

    out_file = 'data/hdnet/merged_train.npz'
    print(f"Saving merged train data to {out_file}...")
    np.savez(out_file, **merged_data)
    print("Done.")

if __name__ == '__main__':
    convert_test_sets()
    merge_and_convert_train_chunks()

