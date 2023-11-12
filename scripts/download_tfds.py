import argparse
import tensorflow_datasets as tfds
import sys

def dataDownloadFromTFDS(dataset_name, data_dir, shuffle_files, splits, download, verbose):
    """
    Downloads a specified dataset from TensorFlow Datasets.

    Args:
        dataset_name (str): Name of the dataset to be downloaded.
        data_dir (str): Directory where the dataset will be downloaded.
        shuffle_files (bool): Whether to shuffle the files during the download.
        splits (list of str): List of dataset splits to download (e.g., ['train', 'validation']).
        download (bool): Flag to enable or disable downloading.
        verbose (bool): If True, provides detailed output during the download process.

    Returns:
        None
    """
    if verbose:
        print(f"Checking availability of [{dataset_name}] in TensorFlow Datasets...")

    # Get all available datasets in TFDS
    datasets_list = tfds.list_builders()

    if dataset_name in datasets_list:
        if verbose:
            print(f"Dataset found. Starting download to {data_dir}...")

        (train_data, test_data), ds_info = tfds.load(
            name=dataset_name,
            split=splits,
            shuffle_files=shuffle_files,
            as_supervised=True,
            with_info=True,
            download=download,
            data_dir=data_dir)
        
        if verbose:
            print("Download completed.")
    else:
        print(f"Dataset {dataset_name} not found in TensorFlow Datasets.")
        sys.exit(1)


if __name__ == "__main__":
    """
    Main entry point of the script. Parses command line arguments and initiates 
    the dataset download process.
    """
    parser = argparse.ArgumentParser(
        prog='data_download',
        description='Downloads dataset from TensorFlow Datasets')
    
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset to download')
    parser.add_argument('--data_dir', type=str, default='../data/',
                        help='Directory to download the data')
    parser.add_argument('--shuffle', action='store_true',
                        help='Whether to shuffle the files during download')
    parser.add_argument('--split', type=str, default='train,validation',
                        help='Dataset splits to download, separated by commas')
    parser.add_argument('--no-download', dest='download', action='store_false',
                        help='Do not download the dataset if it is already available locally')
    parser.set_defaults(download=True)
    parser.add_argument('--verbose', action='store_true',
                        help='Increase output verbosity')

    args = parser.parse_args()

    # Splitting the splits argument into a list
    splits = args.split.split(',')
    dataDownloadFromTFDS(args.dataset, args.data_dir, args.shuffle, splits, args.download, args.verbose)

