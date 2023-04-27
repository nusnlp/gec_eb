import os
import argparse

def main(args):
    candi_path = args.candi_path
    valid_s = os.path.join(candi_path, "valid.src")
    f_valid = open(valid_s).readlines()

    count_path = os.path.join(candi_path, args.file_name)
    count = 0

    with open(count_path, "w+") as f_count:
        for i in range(len(f_valid)):
            content = args.count + "\n"
            count += 1
            f_count.write(content)

    print("Total count: {}".format(count))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--candi_path', type=str, help='*')
    parser.add_argument('--file_name', type=str, default=None,
                        help='*')
    parser.add_argument('--count', type=str, default=5,
                        help='*')
    args = parser.parse_args()
    main(args)