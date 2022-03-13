import tensorflow as tf
from google.protobuf import text_format
def load_protobuf_from_file(filename, proto=tf.GraphDef()):
    with open(filename, 'r') as fdin:
        file_content = fdin.read()
        try:
            graph_def = text_format.Parse(file_content, tf.GraphDef())
            return graph_def
        except text_format.ParseError as e:
            raise IOError("Cannot parse file %s: %s." 
                            % (filename, str(e)))
    return graph_def

def save_protobuf_to_file(protobuf, filename='test_graph.pbtxt'):
    with open(filename, 'w') as fdout:
        fdout.write(text_format.MessageToString(protobuf))

def get_node_by_name(graph_def, node_name):
    for node in graph_def.node:
        if node.name == node_name:
            return node
    return None
    
def fix_graph(graph_filename):
    print('Fix ', graph_filename)
    graph_def = load_protobuf_from_file(graph_filename)

    recv_source_node_dict = {}

    '''
    A -> HVD_AR -> B
    change to 
    A -> B
    '''
    # Find all HorovodAllreduce nodes, record their source node in the dict
    for node in graph_def.node:
        if 'Horovod' not in node.op:
            continue
        if len(node.input)!= 1:
            print('[ERROR] Horovod op has multi input! %s' % node.name)
        node.device = 'cpu:0'
        source_node_name = node.input[0]
        recv_source_node_dict[node.name] = source_node_name

    # Modify the input node name.
    for node in graph_def.node:
        for i in range(len(node.input)):
            input_name = node.input[i]
            if input_name in recv_source_node_dict:
                source_name = recv_source_node_dict[input_name]
                node.input[i] = source_name

    save_protobuf_to_file(graph_def, graph_filename)
    print('Done!')

if __name__ == '__main__':

    for model in model_list:
        graph_filename = '../../ai_simulator/simulator_benchmark/data/graphs/\
                alexnet_bsDefault_gpunum2_partitionGraph/1/graph_0.pbtxt'
        fix_graph(graph_filename)