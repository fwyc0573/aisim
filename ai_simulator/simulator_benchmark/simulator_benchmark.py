# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import yaml
from model_zoo import ModelZoo
from benchmark_tools import BenchmarkTools

# set up the parser
parser = argparse.ArgumentParser(
    prog='python3 simulator_benchmark.py',
    description='Run AI Simulator with 8 models benchmark',
    )

parser.add_argument('--skip-coverage',
                    dest='skip_coverage', action='store_true',
                    help='skip testing the databse coverage')

parser.add_argument('--skip-accuracy',
                    dest='skip_accuracy', action='store_true',
                    help='skip testting the simulator accuracy')

parser.add_argument('-c', '--config_path',
                    dest='config', default='config/config.yaml',
                    help='config setting.')

if __name__ == '__main__':
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    model_zoo = ModelZoo(config)

    models = model_zoo.get_model_list()

    benchmarktools = BenchmarkTools(models,
                                    model_zoo,
                                    args.skip_coverage,
                                    args.skip_accuracy,
                                    config)
    benchmarktools.run()