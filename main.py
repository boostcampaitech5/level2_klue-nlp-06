import yaml
import argparse

from easydict import EasyDict
from src import inference, train, hp_train
from utils import utility

if __name__ == "__main__":
    # run type 설정
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--run_type', type=str, default="both")
    args = parser.parse_args()
    run_type = args.run_type
    print("Run type : ", run_type)

    # config file
    with open('./config.yaml') as f:
        CFG = yaml.load(f, Loader=yaml.FullLoader)

    # code 실행
    if run_type == "inference":
        CFG = EasyDict(CFG)
        inference.main(CFG, run_type, save_path=None)
    else:
        save_path = utility.make_run_name(EasyDict(CFG))
        with open(f'{save_path}/config.yaml', 'w') as f:
            yaml.dump(CFG, f)
        CFG = EasyDict(CFG)

        train.main(CFG, save_path)
        if run_type == "both":
            inference.main(CFG, run_type, save_path)
        