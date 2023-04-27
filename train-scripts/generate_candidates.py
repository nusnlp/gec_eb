import os
import argparse

def main(args):
    root_path = args.root_path
    print(root_path)
    print(args.candidate_name)
    candidate_content = open(os.path.join(root_path, args.candidate_name)).readlines()
    # candidate_content = open("candidate.all").readlines()
    candi_list = []
    
    for candi in candidate_content:
        if candi[0] == "H":
            candi_list.append(candi)

    candi_one_path = os.path.join(root_path, "cand.1")
    candi_two_path = os.path.join(root_path, "cand.2")
    candi_three_path = os.path.join(root_path, "cand.3")
    candi_four_path = os.path.join(root_path, "cand.4")
    candi_five_path = os.path.join(root_path, "cand.5")
    with open(candi_one_path, "w+") as candi_one, \
            open(candi_two_path, "w+") as candi_two, \
            open(candi_three_path, "w+") as candi_three, \
            open(candi_four_path, "w+") as candi_four, \
            open(candi_five_path, "w+") as candi_five:
        for idx in range(len(candi_list)):
            candi_cont = candi_list[idx].split("\t")[-1]
            if idx % 5 == 0:
                candi_one.write(candi_cont)
            elif idx % 5 == 1:
                candi_two.write(candi_cont)
            elif idx % 5 == 2:
                candi_three.write(candi_cont)
            elif idx % 5 == 3:
                candi_four.write(candi_cont)
            elif idx % 5 == 4:
                candi_five.write(candi_cont)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, help='sources to combine')
    parser.add_argument('--candidate_name', type=str, default=None,
                        help='target path if want to use ratio and max length feature')
    args = parser.parse_args()
    main(args)