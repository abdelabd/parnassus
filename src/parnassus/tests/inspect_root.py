import argparse

import uproot


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect ROOT file for matching indices.")
    parser.add_argument("file_path", type=str, help="Path to the ROOT file to inspect.")
    return parser.parse_args()


def main():
    args = parse_args()
    file_path = args.file_path

    with uproot.open(file_path) as f:
        tree = f["Delphes;1"]
        keys = tree.keys()
        print(f"keys: {keys}")

        # print(f"file: {os.path.basename(file_path)}")
        # print("All branches in the file:")
        # for key in sorted(keys):

        #     arr = tree[key].array(library="np")
        #     print(f"    {key} dtype: {arr.dtype}, shape: {arr.shape}, arr[0].shape: {arr[0].shape}")


if __name__ == "__main__":
    main()
