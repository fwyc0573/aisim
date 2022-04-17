# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from superscaler.ai_simulator.simulator.device import Device


class FIFODevice(Device):
    def __init__(self, name):
        super().__init__(name)
        # The reference of enqueued nodes.
        self.__node_queue = []
        # Head pointer of node queue
        self.__queue_head = 0

        self.__overlapped = False

    # Whether the device is idle.
    def add_counter(self, counter, mul):
        self._mul = mul
        self._counter = counter

    # Whether the device is idle.
    def is_idle(self):
        return self.__queue_head == len(self.__node_queue)

    # Return the ref of current executing node
    def get_next_node(self):
        return self.__node_queue[self.__queue_head]

    # Enqueue a node to this device.
    # If the node is idle, update the head_end_time.
    def enqueue_node(self, node, time_now):
        if self.is_idle():
            if not self._counter.is_idle():
                node.set_execution_time(node.get_execution_time()*self._mul)
            self._next_finish_time = time_now + node.get_execution_time()
        self.__node_queue.append(node)

    # Dequeue a node in this device.
    # If still has node in queue, reset head end time
    def dequeue_node(self):
        self.__queue_head += 1
        if not self.is_idle():
            head_node = self.__node_queue[self.__queue_head]
            head_node.set_execution_time(head_node.get_execution_time()*self._mul)
            next_node_timeuse = head_node.get_execution_time()
            self._next_finish_time += next_node_timeuse

    # def refresh_node(self, time_now):
    #     if not self.is_idle():
    #         head_node = self.__node_queue[self.__queue_head]
    #         self._next_finish_time = head_node._enqueue_time + head_node.get_execution_time()

