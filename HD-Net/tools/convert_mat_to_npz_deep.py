import os
import glob
import numpy as np
import h5py

def parse_hdf5_group(f, group):
    """
    Extracts datasets from an HDF5 group, correctly handling MATLAB's column-major data
    (transposing it) and resolving object references (like strings or Cell Arrays).
    """
    data_dict = {}
    for key in group.keys():
        if key == '#refs#':
            continue

        dataset = group[key]
        if isinstance(dataset, h5py.Dataset):

            if dataset.dtype == 'object':

                refs = dataset[()].flatten()
                parsed_strings = []
                for ref in refs:
                    target = f[ref][()]


                    try:
                        chars = ''.join(chr(c) for c in target.flatten())
                        parsed_strings.append(chars)
                    except:
                        parsed_strings.append(str(target))


                data_dict[key] = np.array(parsed_strings)
            else:





                val = np.array(dataset).T
                data_dict[key] = val

    return data_dict

def convert_test_sets():
    base_dir = 'data/hdnet'

    test_files = [
        'test_calbased.mat',
        'test_calfree.mat'
    ]

    for filename in test_files:
        filepath = os.path.join(base_dir, filename)
        if os.path.exists(filepath):
            print(f"Reading and flattening {filepath}...")
            with h5py.File(filepath, 'r') as f:

                if 'Subset' in f:
                    data = parse_hdf5_group(f, f['Subset'])
                else:
                    data = parse_hdf5_group(f, f)

            out_path = filepath.replace('.mat', '.npz')
            print(f"Saving to {out_path}...")
            np.savez(out_path, **data)
            print(f"Shapes saved for {filename}:")
            for k, v in data.items():
                print(f"  {k}: {v.shape} ({v.dtype})")
            print("Done.\n")

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
        with h5py.File(cf, 'r') as f:
            if 'Subset' in f:
                data = parse_hdf5_group(f, f['Subset'])
            else:
                data = parse_hdf5_group(f, f)

            for k, v in data.items():
                if k not in merged_data:
                    merged_data[k] = []
                merged_data[k].append(v)

    print("\nConcatenating chunks along axis 0...")
    final_data = {}
    for k in merged_data.keys():
        try:
            final_data[k] = np.concatenate(merged_data[k], axis=0)
            print(f"  {k}: successfully concatenated to shape {final_data[k].shape}")
        except Exception as e:
            print(f"  Failed to concatenate key {k}: {e}. Skipping.")

    out_file = 'data/hdnet/merged_train.npz'
    print(f"\nSaving merged train data to {out_file}...")
    np.savez(out_file, **final_data)
    print("Done.")

if __name__ == '__main__':
    convert_test_sets()
    merge_and_convert_train_chunks()

