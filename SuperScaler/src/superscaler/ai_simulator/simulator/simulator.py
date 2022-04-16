# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

'''
Author: v-hohua
Version: 0.1
Update date: 12/16/2019

Simulation Execution module of AI Simulator project.
This module will read execution graph, then simulate the execution.

v1.0 supported function:
Asynchronized execution on multiple devices.

v1.0 execute required node attributes:
Defined in node_define.py
'''

from superscaler.ai_simulator.simulator.node import NodeMetadata
from superscaler.ai_simulator.simulator.node import Node
from superscaler.ai_simulator.simulator.device import Device
from superscaler.ai_simulator.simulator.device_factory import DeviceFactory
from superscaler.ai_simulator.simulator.tensor import Tensor
from superscaler.ai_simulator.simulator.fifo_device import FIFODevice
from sortedcontainers import SortedSet
import warnings
import copy

RET_SIMULATION_FINISH = -1
NCCL_MUL = 2.2
COMP_MUL = 1.3

class Simulator():
    def __init__(self, nodemetadata_list, device_info):
        '''Init Simulator with nodemetadata namedtuples and a device list

        Args:
            nodemetadata_list: a list of namedtuple or dict, storing
                nodemetadata
            device_info: a list of tuple (device_type, spec_list) containing
                device info, or a list of class Device, storing all Device
        '''
        # List of NodeMetadata. All nodes in the graph
        self.__nodes_metadata = []
        # All nodes.
        self.__nodes = []
        # All devices.
        self.__devices = {}
        self.__sorted_Device_set = SortedSet()

        # The execution result of all nodes
        self.__execution_enqueue_time = []  # (node_index, node enqueue time)
        self.__execution_dequeue_time = []  # (node_index, node dequeue time)
        # Current simulation timestamp.
        self.__time_now = 0.0

        # Init all node metadata
        self.__init_node_metadata(nodemetadata_list)
        # Init devices list
        self.__init_device(device_info)

        self.__network_device = None
        self.__compute_device = None

        # Check the coherence of nodemetadata_list and device_info
        for node_metadata in self.__nodes_metadata:
            device_name = node_metadata.device_name
            if device_name not in self.__devices:
                raise TypeError(
                    device_name + " in nodemetadata_list doesn't exist on"
                    + " device_info")
            new_node = Node(node_metadata, self.__devices[device_name])
            self.__nodes.append(new_node)

            if 'GPU' in device_name:
                self.__compute_device = device_name
            if 'CPU' in device_name:
                self.__network_device = device_name

        # Init edges in nodes
        for node in self.__nodes:
            node.renew_successor_nodes(self.__nodes)

    def __check_counter_part(self, device_name):
        if device_name == self.__compute_device:
            if self.__network_device != None:
                counter_device_name = self.__network_device
            else:
                return False
        if device_name == self.__network_device:
            if self.__compute_device != None:
                counter_device_name = self.__compute_device
            else:
                return False
        device = self.__devices[counter_device_name]
        if device.is_idle():
            return False
        return True

    def __start_all_ready_nodes(self):
        '''Start all node in ready list.
        Only call once at the beginning of a simulation.
        '''
        for node in self.__nodes:
            if node.is_ready():
                self.__start_node_with_SortedSet(node)

    def __start_node(self, exec_node):
        '''Start to execute a node.Enqueue the node into device.
        The node will be marked as 'executing'.

        @param exec_node:    Node ref. The node to execute.
        '''
        node_id = exec_node.get_index()
        self.__execution_enqueue_time.append((node_id, self.__time_now))
        exec_node.execute(self.__time_now)

    def __find_earliest_complete_device(self):
        earliest_complete_time = RET_SIMULATION_FINISH
        earliest_device = None
        for device_name in self.__devices:
            device = self.__devices[device_name]
            if device.is_idle():
                continue
            device_complete_time = device.get_next_finish_time()
            if device_complete_time < earliest_complete_time or \
                    earliest_complete_time == RET_SIMULATION_FINISH:
                earliest_complete_time = device_complete_time
                earliest_device = device
        return earliest_complete_time, earliest_device

    def __start_node_with_SortedSet(self, exec_node):
        '''Start to execute a node.Enqueue the node into device.
        Also add device with first node into __sorted_Device_set.
        The node will be marked as 'executing'.

        @param exec_node:    Node ref. The node to execute.
        '''
        device_name = exec_node.get_device_name()
        device = self.__devices[device_name]

        if not isinstance(device, FIFODevice) and device in self.__sorted_Device_set:
            self.__sorted_Device_set.remove(device)

        if self.__check_counter_part(device_name):
            if device_name == self.__compute_device:
                exec_node.set_execution_time(exec_node.get_execution_time() * COMP_MUL)
            if device_name == self.__network_device:
                exec_node.set_execution_time(exec_node.get_execution_time() * NCCL_MUL)


        node_id = exec_node.get_index()
        self.__execution_enqueue_time.append((node_id, self.__time_now))
        exec_node.execute(self.__time_now)

        if device not in self.__sorted_Device_set:
            self.__sorted_Device_set.add(device)

    def __find_earliest_complete_device_by_SortedSet(self):
        earliest_complete_time = RET_SIMULATION_FINISH
        earliest_device = None

        is_fifo_device = False
        
        # find earliest complete device by SortedSet
        # All next_finish_time of FIFOdevice are stored on self.__sorted_Device_set
        # As those values are changeless, we can use a sorted set to store them,
        # Then we can find the earliest complete device on O(log(n))
        if len(self.__sorted_Device_set) > 0:
            earliest_device = self.__sorted_Device_set[0]
            earliest_complete_time = earliest_device.get_next_finish_time()
            self.__sorted_Device_set.pop(0)

        return earliest_complete_time, earliest_device

    def __next_step(self):
        '''Wait until any executing node is done. Get the timestamp.
        Mark the node as 'done'. Then dequeue it from device.
        Update all successor nodes' dependency counter.
        If a successor node is ready, start it.
        '''
        # Find the first completed node
        earliest_complete_time, earliest_device = \
            self.__find_earliest_complete_device_by_SortedSet()

        if earliest_complete_time == RET_SIMULATION_FINISH:
            return RET_SIMULATION_FINISH

        self.__time_now = earliest_complete_time
        earliest_node = earliest_device.get_next_node()
        # Handle the node
        earliest_node.finish()
        self.__execution_dequeue_time.append((earliest_node.get_index(),
                                              self.__time_now))

        # Insert the next node into self.__sorted_Device_set
        if not earliest_device.is_idle():
            self.__sorted_Device_set.add(earliest_device)

        # Handle successor nodes
        for suc_node in earliest_node.get_successor_nodes():
            suc_node.decrease_remain_dependency_cnt(1)
            if suc_node.is_ready():
                self.__start_node_with_SortedSet(suc_node)

        device_name = earliest_device.name()
        if self.__check_counter_part(device_name) and earliest_device.is_idle():
            if device_name == self.__compute_device:
                counter_device = self.__devices[self.__network_device]
                counter_device._next_finish_time = (counter_device._next_finish_time - self.__time_now) / NCCL_MUL + self.__time_now
            if device_name == self.__network_device:
                counter_device = self.__devices[self.__compute_device]
                counter_device._next_finish_time = (counter_device._next_finish_time - self.__time_now) / COMP_MUL + self.__time_now

        return earliest_complete_time

    def reset(self):
        '''Reset the simulator'''
        self.__time_now = 0.0
        self.__execution_enqueue_time = []
        self.__execution_dequeue_time = []
        for node in self.__nodes:
            node.reset()

    def list_undone_nodes(self):
        '''Return list of undone nodes to check if all nodes have been executed
        '''
        undone_nodes = []
        for node in self.__nodes:
            if not node.is_done():
                undone_nodes.append(node)
        return undone_nodes

    def __check_if_all_nodes_done(self):
        '''Send a warining if there are nodes haven't been done. Only show the
        top 10 nodes' detail information.
        the Warnings are like this:
        There are 1155/1589 nodes haven't been executed:
          Index:434    Name:conv0/batchnorm0/FusedBatchNorm
          Index:435    Name:conv0/Relu
          ...
        '''
        undone_nodes = self.list_undone_nodes()
        if undone_nodes:
            not_done_nodes = ''
            not_done_num = len(undone_nodes)
            nodes_tmp = '\n  Index:%-6d Name:%s  '
            warn_tmp = "\nThere are %d/%d nodes haven't been executed:%s"
            for node in undone_nodes[:10]:
                not_done_nodes += nodes_tmp \
                    % (node.get_index(),
                       node.get_name())
            if not_done_num > 10:
                not_done_nodes += '\n  ...'
            warnings.warn(warn_tmp % (not_done_num,
                                      len(self.__nodes),
                                      not_done_nodes))

    def run(self):
        '''Run the simulation'''
        self.reset()
        finish_time = 0.0
        # Enqueue all ready nodes.
        self.__start_all_ready_nodes()
        while(finish_time != RET_SIMULATION_FINISH):
            # Wait until one node is done.
            finish_time = self.__next_step()
        # send warning if there are nodes haven't been executed
        self.__check_if_all_nodes_done()
        return (self.__time_now,
                self.__execution_enqueue_time,
                self.__execution_dequeue_time)

    def get_nodes(self):
        '''Get the all nodes'''
        return self.__nodes

    def __init_node_metadata(self, nodemetadata_list):
        '''Init all NodeMetadata based on nodemetadata_list, which could be a
        list of namedtuple or a list of dict.
        '''
        if not isinstance(nodemetadata_list, list):
            raise ValueError("Input nodemetadata_list should be a list.")
        elif len(nodemetadata_list) == 0:
            return
        else:
            for node in nodemetadata_list:
                if isinstance(node, tuple):
                    node = dict(node._asdict())
                if isinstance(node, dict):
                    node = self.__init_output_tensors(node)
                    metadata = NodeMetadata(
                        index=node['index'],
                        op=node['op'],
                        name=node['name'],
                        device_name=node['device_name'],
                        execution_time=node['execution_time'],
                        output_tensors=node['output_tensors'],
                        input_ids=node['input_ids'],
                        dependency_ids=node['dependency_ids'],
                        successor_ids=node['successor_ids']
                    )
                    self.__nodes_metadata.append(metadata)
                else:
                    raise ValueError("Invalid Input nodemetadata_list.")

    def __init_device(self, device_info):
        '''Init self.__devices via device_info
        '''
        if not isinstance(device_info, list):
            raise ValueError("Input device_info should be a list.")
        if isinstance(device_info[0], Device):
            # device_info is a list of class Device
            for device in device_info:
                self.__devices[device.name()] = device
        elif isinstance(device_info[0], tuple):
            device_factory = DeviceFactory()
            # device_info is a list of (type, spec) tuple
            for device_type, spec_list in device_info:
                device = device_factory.generate_device(
                    device_type, *spec_list)
                self.__devices[device.name()] = device
        else:
            raise ValueError("Invalid input device_info parameter.")

    def __init_output_tensors(self, node):
        '''Add output_tensors attribute to node
        '''
        # Add output_tensors attributes
        node['output_tensors'] = []
        if node['op'] == 'Send' or node['op'] == 'Recv':
            # Set device_name to NetworkSimulator for send/recv nodes
            if node['op'] == 'Send':
                # Check whether tensor_type is valid
                if not Tensor.check_tensor_type(node['tensor_type']):
                    raise ValueError("Node have invalid tensor_type")
                # Add Tensor list
                node['output_tensors'] = [
                    Tensor(node['tensor_type'], node['size'])]
        return node
