import os
import json
import argparse
import yaml
from ai_simulator.simulator_benchmark.model_zoo import ModelZoo

# set up the parser
parser = argparse.ArgumentParser(
    prog='python3 simulator_benchmark.py',
    description='Run AI Simulator with 8 models benchmark',
    )

parser.add_argument('-c', '--config_path',
                    dest='config', default='config_torch.yaml',
                    help='config setting.')
parser.add_argument("--local_rank", default=-1, type=int)
parser.add_argument("--batchsize", default=32, type=int)
parser.add_argument('--model', type=str, default='resnet50',
                    help='model to benchmark')

ddp_meta_command = "pdsh -N -R ssh -w ^host2 'sudo docker exec --workdir /mnt/aisim lc bash /mnt/aisim/ddp_profile.sh {} {} {} {}'"

def baseline_test(args, config):
    for model in args.model_list:
        print(model, args.model_list)
        for _, value in config['enviroments'].items():
            node = value//8
            cmd = ddp_meta_command.format(node,
                                          args.model_zoo.get_sub_models(model), 
                                          args.model_zoo.get_batch_size(model),
                                          args.model_zoo.get_type(model))
            time = os.popen(cmd).read().split('\n')[0]
            args.model_zoo.set_baseline_time(value, model, float(time))
    args.model_zoo.dump_baseline()
    print(args.model_zoo.get_baseline())

def one_click_test(args, config):

    baseline_test(args, config)

if __name__ == '__main__':
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f)

    model_zoo = ModelZoo(config)
    args.model_zoo = model_zoo

    model_list = model_zoo.get_model_list()
    args.model_list = model_list

    print(model_list)
    print(config)

    one_click_test(args, config)

