import os
import time
from .models import tf_model
import tensorflow as tf
from tensorflow.python.client import timeline as _timeline
from tensorflow.python.profiler import model_analyzer
import horovod.tensorflow as hvd
from google.protobuf import text_format
tf.set_random_seed(1)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'


def output_csv(filename, csv_list, path='./', scale=1):
    # TODO: description
    # output the csv of time list or loss list
    filepath = path + '/' + filename
    with open(filepath, mode='w') as f:
        for i in range(len(csv_list[0])):
            first_column = True
            for j in range(len(csv_list)):
                write_str = str(csv_list[j][i]*scale)
                if first_column:
                    first_column = False
                else:
                    write_str = ',' + write_str
                f.write(write_str)
            f.write('\n')
        f.close()


def generate_tfprof_profile(profiler, tfprof_file, print_ops=10):
    # Generates a tfprof profile, writing it to a file and printing top ops.
    profile_proto = profiler.serialize_to_string()
    print('Dumping ProfileProto to %s' % tfprof_file)
    with open(tfprof_file, 'wb') as f:
        f.write(profile_proto)

    # Print out the execution times of the top operations.
    if print_ops > 0:
        options = tf.profiler.ProfileOptionBuilder.time_and_memory()
        options['max_depth'] = print_ops
        options['order_by'] = 'accelerator_micros'
        profiler.profile_operations(options)


def save_graph(graph_def, path='./', filename='graph'):
    tf.io.write_graph(graph_def, path, filename + '.pbtxt')


def save_partition_graph(metadata, path='./', filename='graph'):
    filepath = os.path.join(path, filename)
    for count, partition in enumerate(metadata.partition_graphs):
        graph_txt = text_format.MessageToString(partition)

        with open('%s_%d.pbtxt' % (filepath, count), 'w') as file:
            file.write(graph_txt)


def get_node_by_name(graph_def, node_name):
    for node in graph_def.node:
        if node.name == node_name:
            return node
    return None


def fix_graph_sendrecv(graph_def):
    '''
    # Change recv node to NoOp, change name to dummy name
    node.op = 'NoOp'
    raw_recv_op_name = node.name
    node.name = 'Dummy_%s' % node.name
    # Change send node to NoOp, and redirect input as dependent input to recv node
    send_node.input[0] = '^' + node.name
    # Change source node's name to recv node's name
    source_node.name = raw_recv_op_name
    '''
    recv_source_node_dict = {}
    # Find all recv nodes, record their source node in the dict
    for node in graph_def.node:
        if 'Recv' not in node.op:
            continue
        if len(node.input)!= 1:
            print('[ERROR] Recv op has multi input! %s' % node.name)
        send_node_name = node.input[0].replace('^','')
        send_node = get_node_by_name(graph_def, send_node_name)
        if len(send_node.input)!= 1:
            print('[ERROR] Send op has multi input! %s' % send_node.name)
        source_node_name = send_node.input[0]

        recv_source_node_dict[node.name] = source_node_name

    # Modify the input node name.
    for node in graph_def.node:
        for i in range(len(node.input)):
            input_name = node.input[i]
            if input_name in recv_source_node_dict:
                source_name = recv_source_node_dict[input_name]
                node.input[i] = source_name

    return graph_def


def save_partition_graph_shapes(metadata, path='./', filename='graph'):
    for count, partition in enumerate(metadata.partition_graphs):
        graph_txt = text_format.MessageToString(partition)

        graph_def = text_format.Parse(graph_txt,
                                      tf.compat.v1.GraphDef())
        graph_def = fix_graph_sendrecv(graph_def)
        graph_clone = tf.Graph()
        with graph_clone.as_default():
            tf.import_graph_def(graph_def=graph_def, name="")
            shaped_graph = graph_clone.as_graph_def(add_shapes=True)
            tf.io.write_graph(shaped_graph, path,
                              '%s_%d.pbtxt' % (filename, count))


def run_model(model, horovod=False, gpu_num=1, output=None,
              steptime=False, profile=False, timeline=False, loss=False,
              session=1, step=1, batchsize=None, graph=False):
    # TODO: description

    # cannot dump graph if timeline or profile is On
    if graph and (timeline or profile):
        raise ValueError("cannot dump graph togother with timeline or tfprof")

    with tf.Graph().as_default():

        times_list = []
        losses_list = []
        op, _loss = tf_model.get_model(model, batchsize, horovod=horovod)

        # set gpus available
        config = tf.ConfigProto()
        if horovod is True:
            config.gpu_options.allow_growth = False
            config.gpu_options.visible_device_list = str(hvd.local_rank())
            # print('DEBUG: ', str(hvd.local_rank()))
        else:
            # buildup gpus='0,1,2...'
            config.gpu_options.allow_growth = False
            gpus = ','.join(map(str, range(gpu_num)))
            print('DEBUG: gpus=%s' % gpus)
            config.gpu_options.visible_device_list = gpus

        for i in range(session):

            sess = tf.Session(config=config)
            sess.run(tf.global_variables_initializer())
            times = []
            losses = []

            opts = None
            run_metadata = None

            # the dump graph mode on
            if graph:
                opts = tf.RunOptions(output_partition_graphs=True)
                run_metadata = tf.RunMetadata()
            # the profile mode on
            elif profile or timeline:
                # create runOptions and run_metadata object
                opts = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()

                if profile:
                    # Create a profiler.
                    profiler = model_analyzer.Profiler(sess.graph)
            for n in range(step):
                start_time = time.time()

                # run model
                if loss is True:
                    res = sess.run([op, _loss], options=opts,
                                   run_metadata=run_metadata)
                    losses.append(res[1])
                else:
                    res = sess.run(op, options=opts,
                                   run_metadata=run_metadata)

                train_time = time.time() - start_time
                times.append(train_time)

                # print steptime and loss at realtime
                if loss is True:
                    print('Sess%d/%d Step%d/%d: time=%.2fms loss=%.2f' %
                          (i+1, session, n+1, step, train_time*1000, res[1]))
                else:
                    print('Sess%d/%d Step%d/%d: time=%.2fms' %
                          (i+1, session, n+1, step, train_time*1000))
                if (not graph) and profile:
                    profiler.add_step(step=step, run_meta=run_metadata)

            times_list.append(times)
            losses_list.append(losses)

        if output is not None:

            # make folder if it not exist
            try:
                if not os.path.exists(output):
                    os.makedirs(output)
            except(FileExistsError):
                print("")

            file_loss = '_lossOn' if loss else ''
            file_trace = '_traceOn' if profile or timeline else ''
            file_horovod = '_hvdRank%d' % hvd.rank() if horovod else ''
            file_batchsize = '_bs%d' % batchsize if batchsize is not None\
                else '_bsDefault'
            file_gpunum = '_gpunum%d' % gpu_num

            if steptime is True:
                filename = '%s%s%s%s%s%s_steptime.csv' %\
                    (model, file_batchsize, file_loss, file_trace,
                        file_horovod, file_gpunum)
                output_csv(filename, times_list, path=output, scale=1000)

            if loss is True:
                filename = '%s%s%s%s%s%s_loss.csv' % \
                    (model, file_batchsize, file_loss, file_trace,
                        file_horovod, file_gpunum)
                output_csv(filename, losses_list, path=output)

            if graph:
                # save each partition of graph with _output_shapes attr

                if horovod:
                    graph_dir = os.path.join(output,
                                             '%s%s%s%s_partitionGraph'
                                             % (model, file_batchsize,
                                                file_loss,
                                                file_gpunum),
                                             str(hvd.rank()))
                    if not os.path.exists(graph_dir):
                        os.makedirs(graph_dir)
                    save_partition_graph_shapes(run_metadata,
                                                graph_dir, 'graph')
                else:
                    save_partition_graph_shapes(run_metadata, output,
                                                '%s%s%s%s%s_partitionGraph'
                                                % (model, file_batchsize,
                                                    file_loss,
                                                    file_horovod,
                                                    file_gpunum))

            if profile is True:
                filename = '%s%s%s%s%s_gpunum%d.profile' % \
                    (model, file_batchsize, file_loss, file_trace,
                        file_horovod, gpu_num)
                filepath = output + '/' + filename
                generate_tfprof_profile(profiler, filepath)

            if timeline is True:
                filename = '%s%s%s%s%s_gpunum%d.timeline' % \
                    (model, file_batchsize, file_loss, file_trace,
                        file_horovod, gpu_num)
                filepath = output + '/' + filename
                tl = _timeline.Timeline(run_metadata.step_stats)
                ctf = tl.generate_chrome_trace_format()
                with open(filepath, 'w') as f:
                    f.write(ctf)
