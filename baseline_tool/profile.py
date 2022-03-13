import sys
import os
import argparse

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import tensorflow as tf # noqa E402
tf.contrib._warning = None # noqa E402

import horovod.tensorflow as hvd # noqa E402
from tf_profile.models import tf_model # noqa E402
from tf_profile import tf_profile # noqa E402

# set up the parser
parser = argparse.ArgumentParser(
    prog='python3 profile.py',
    description='Run and do profiling to some tensorflow models',
    )

# add args to the parser
parser.add_argument('--list', dest='list', action='store_true',
                    help='list out all the models available')

parser.add_argument('-m', dest='models', nargs='*', default=['all'],
                    help='select which model(s) to run. (default all)')

parser.add_argument('--batchsize', dest='batchsize', nargs='?',
                    type=int, default=None,
                    help='set the number of models\' batchsize \
                        (each model have unique default batchsize,\
                            check it in README)')

parser.add_argument('-n', dest='gpu_num', nargs='?', type=int, default=1,
                    help='set number of GPUs to run')

parser.add_argument('--horovod', dest='horovod', action='store_true',
                    help='use horovod modified model if this arg added')

parser.add_argument('-o', dest='output_folder', nargs='?',
                    type=str, default='./',
                    help='set the path of the output\'s \
                        folder (default -o ./)')

parser.add_argument('-g', '--graph', dest='graph', action='store_true',
                    help='output the model.pbtxt if this arg added')

parser.add_argument('-t', '--time', dest='steptime', action='store_true',
                    help='output the steptime.csv if this arg added')

parser.add_argument('-l', '--loss', dest='loss', action='store_true',
                    help='output the loss.csv if this arg added')

parser.add_argument('-p', '--tfprof', dest='profile', action='store_true',
                    help='output the model.profile which is the input of\
                         tfprof if this arg added')

parser.add_argument('--time-line', dest='timeline', action='store_true',
                    help='output the model.timeline which is the input\
                         of tensorflow timeline if this arg added')

parser.add_argument('--session', dest='session_num', nargs='?',
                    type=int, default=10,
                    help='set the number of session to be run for each model \
                        (default 10)')

parser.add_argument('--step', dest='step_num', nargs='?',
                    type=int, default=600,
                    help='set the number of steps to be run for each session \
                        (default 600)')

parser.add_argument('--version', '-v', action='version', version='v0.1.0')


if __name__ == '__main__':
    args = parser.parse_args()

    # if --list, print out the list of models
    if args.list is True:
        models = tf_model.get_model_list(horovod=args.horovod)
        print('All of the models with horovod=%s:' % str(args.horovod))
        for model in models:
            print('  %s' % model)
        exit()

    # check if the models available
    models = []
    for i in args.models:
        if i == 'all':
            models = tf_model.get_model_list(horovod=args.horovod)
        elif tf_model.exist_model(i, horovod=args.horovod):
            models.append(i)
        else:
            raise ValueError('model: %s doesn\'t exist' % i)

    # if --graph, generate the pbtxt graph od models

    if args.graph is True:
        if args.profile:
            print('the --graph cannot run together with --tfprof')
            args.profile = False
        if args.timeline:
            print('the --graph cannot run together with --timeline')
            args.timeline = False
        if args.session_num != 1:
            print('because --graph setted, session_num auto set to 1')
            args.session_num = 1
        if args.step_num != 1:
            print('because --graph setted, step_num auto set to 1')
            args.step_num = 1

    # if we don't need the benchmark tool to run, set --session 0
    if args.session_num <= 0:
        exit()

    # decide if use horovodrun to run again
    print('DEBUG: OMPI_COMM_WORLD_RANK=%s'
          % os.environ.get('OMPI_COMM_WORLD_RANK'))
    if args.horovod and os.environ.get('OMPI_COMM_WORLD_RANK') is None:
        # buildup the command to run horovod
        command = 'horovodrun -np %d -H localhost:%d python3' \
            % (args.gpu_num, args.gpu_num)
        for arg in sys.argv:
            command = command + ' ' + arg
        print('DEBUG: auto-run with horovodrun:\n  %s' % command)
        os.system(command)
        exit()

    # detect if auto set horovod=True
    horovod = args.horovod
    gpu_num = args.gpu_num
    if os.environ.get('OMPI_COMM_WORLD_RANK') is not None:
        print('DEBUG: MPI detected, run with horovod model')
        hvd.init()
        horovod = True
        gpu_num = hvd.size()

    print('DEBUG: horovod=%s, gpu_num=%s' % (horovod, gpu_num))

    for model in models:
        print('Run model %s, batchsize=%s, horovod=%s, gpu_num=%d' %
              (model, str(args.batchsize), str(horovod), gpu_num))
        tf_profile.run_model(model, horovod=horovod, gpu_num=gpu_num,
                             output=args.output_folder, steptime=args.steptime,
                             profile=args.profile, timeline=args.timeline,
                             loss=args.loss, session=args.session_num,
                             step=args.step_num, batchsize=args.batchsize,
                             graph=args.graph)
