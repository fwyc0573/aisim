# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from superscaler.ai_simulator.simulator.network_simulator.flow import Flow
from superscaler.ai_simulator.simulator.network_simulator.link_manager import \
     LinkManager
from superscaler.ai_simulator.simulator.device import Device
from sortedcontainers import SortedSet


class NetworkSimulator(Device):
    link_essential_data = {'link_id': int, 'source_name': str,
                           'dest_name': str}
    link_extra_data = {'capacity': str, 'latency': str}

    def __init__(self, name, links_spec, routing_info):
        '''Init NetworkSimulator with links (to generate topology) and routing
        info

        Args:
            links_spec: list of dict, [{'link_id': int, 'source_name': str,
                                        'dest_name': str, 'capacity': str,
                                        'latency': str}]
                src/dest_name format:
                    /server/hostname/DeviceType/DeviceIndex/
                    /switch/switch_name/
            routing_info: {(src_name, dest_name, path_index):[id0, id1..]}
        '''
        super().__init__(name)
        self.__flows = SortedSet()
        # Init links
        self.__link_manager = LinkManager(links_spec, routing_info)

    def is_idle(self):
        '''Check whether the network simulator is idle
        '''
        return len(self.__flows) == 0

    def get_next_node(self):
        '''Return (next finish node, estimated finish time)
        '''
        if self.is_idle():
            return None
        next_flow = self.__flows[0]
        return next_flow.node

    def enqueue_node(self, node, time_now):
        '''Enqueue a node and update all the flows. The node will first be
        turned into a Flow, then add to the list, and update all Flow's
        capacities

        Args:
            node: class Node, a Send op
                node.name format: ":send:src_name:dest_name:path_index:"
                ":send:/server/hostname1/GPU/0/:/server/hostname1/GPU/1/:0:"
            time_now: float, current time of simulation
        '''
        # Check whether there are flows need to be dequeued first
        if not self.is_idle() and self.get_next_finish_time() < time_now:
            raise ValueError(
                "There are flows need to be dequeued before "
                "enqueuing a new flow!")
        if node.get_op() == 'Send':
            # Enqueue and calculate Send nodes' communication cost
            new_flow = Flow(node, time_now)
            self.__flows.add(new_flow)
            # Add flow to all related links
            route_path = self.__link_manager.get_routing(
                new_flow.node.get_name())
            if route_path is None:
                raise ValueError(
                    "No routing info for current Node: {0})".format(
                        new_flow.node.get_index()))
            new_flow.set_routing_info(route_path)
            for link in route_path:
                link.add_flow(new_flow)
            # Calculate and update all flows' current capacities
            self.__update_all_flows_capacities(time_now, new_flow)
        elif node.get_op() == "Recv":
            # Recv nodes only act as a dependency. Therefore, Recv nodes'
            # output_tensors should be []
            if node.get_tensors() != []:
                raise ValueError(
                    "Invalid op='Recv' Node, node index {0}".format(
                        node.get_index()
                    ))
            new_flow = Flow(node, time_now)
            # Refresh the estimated_finish_time of the flow, the available
            # bandwith could be any number larger than 0
            new_flow.set_available_bandwidth(float('inf'), time_now)
            self.__flows.add(new_flow)
        self._next_finish_time = self.__flows[0].get_estimated_finish_time()

    def dequeue_node(self):
        '''Dequeue the flow with the smallest estimated_finish_time'''
        if self.is_idle():
            return
        next_flow = self.__flows.pop(0)
        if next_flow.node.get_op() == "Send":
            # Dequeue a Send Node
            # Delete next_flow in all related links
            route_path = self.__link_manager.get_routing(
                next_flow.node.get_name())
            if route_path is None:
                raise ValueError(
                    "No routing info for current Node: {0})".format(
                        next_flow.node.get_index()))
            for link in route_path:
                link.delete_flow(next_flow)
            time_now = next_flow.get_estimated_finish_time()
            # Calculate and update all flows' current capacities
            self.__update_all_flows_capacities(time_now, next_flow)
        elif next_flow.node.get_op() == "Recv":
            # Dequeue a Recv Node
            pass
        if self.is_idle():
            self._next_finish_time = -1
        else:
            self._next_finish_time = self.__flows[0].get_estimated_finish_time()

    def __update_affected_range(self, flow):
        '''
        Return two lists. Affected flows and link_ids.
        '''
        affected_flows = set([])
        affected_links = set([])
        new_path = flow.get_routing_info()
        new_links = set()
        new_flows = set()
        if len(new_path) > 0:
            update_flag = True
            for link_id in new_path:
                new_links.add(link_id)
        while(update_flag):
            update_flag = False
            # Check all links of new_links, add flows in them to new_flow
            for link_id in new_links:
                link = self.__link_manager.get_link(link_id)
                for flow in link.flows:
                    if flow not in affected_flows:
                        new_flows.add(flow)
                        update_flag = True
                affected_links.add(link_id)
            new_links.clear()
            # deal with all new_flows
            for flow in new_flows:
                flow_path = flow.get_routing_info()
                for link_id in flow_path:
                    if link_id not in affected_links:
                        new_links.add(link_id)
                        update_flag = True
                affected_flows.add(flow)
            new_flows.clear()
        return affected_flows, affected_links

    def __update_all_flows_capacities(self, time_now, update_flow):
        '''
        update_flow: which flow's status updating triggers the function.
        Used to determine affected links and flows.

        Calculate all flow's capacities, and update flows' status during
        [last_start_time, time_now)

        There are two events that will change the flows' capacities: enqueue
        and dequeue.

        '''
        unfinished_schedule_flow = set()
        flow_current_capacity = {}  # {flow_obj: capacity}
        flow_capacity_log = {}  # {flow_obj: [capacities]}

        affected_flows, affected_links = self.__update_affected_range(update_flow)

        if len(affected_flows) == 0:
            # It is possilbe that there are affected links but no affected flow.
            # When the updated flow is finished and it does not share links with other flows.
            # Then do not need to run updating algorithm.
            return

        # Init variables
        for flow in affected_flows:
            if flow.node.get_op() != "Send":
                # Only process Send Node
                continue
            # At first, each flow get 0 BW
            flow_current_capacity[flow] = 0
            flow_capacity_log[flow] = []
            unfinished_schedule_flow.add(flow)
        # Iterate to assign all flows' capacity/bandwidth
        while(len(unfinished_schedule_flow) != 0):
            # Schedule all links
            for link_id in affected_links:
                link = self.__link_manager.get_link(link_id)
                if len(link.flows) == 0 :
                    continue
                link_schedule_result = self.__schedule_link(
                    link,
                    flow_current_capacity,
                    unfinished_schedule_flow)
                # Log all flows capacity to a smaller one
                for flow_in_link, flow_capacity in \
                        link_schedule_result.items():
                    flow_capacity_log[flow_in_link].append(flow_capacity)
            # Log which flow is the bottleneck
            bottleneck_flow = None
            bottleneck_capacity = float('inf')
            for flow in flow_current_capacity:
                if flow not in unfinished_schedule_flow:
                    continue
                # Use minimum as its capacity
                flow_current_capacity[flow] = min(flow_capacity_log[flow])
                # Reset flow_capacity_logs
                flow_capacity_log[flow] = []
                if flow_current_capacity[flow] < bottleneck_capacity:
                    bottleneck_capacity = flow_current_capacity[flow]
                    bottleneck_flow = flow

            if bottleneck_flow is not None:
                # bottleneck_flow is finished, remove it from set.
                unfinished_schedule_flow.remove(bottleneck_flow)

        # Update bandwidth, and change flow status
        for flow in flow_current_capacity:
            self.__flows.discard(flow)
            flow.set_available_bandwidth(flow_current_capacity[flow], time_now)
            self.__flows.add(flow)

    def __schedule_link(self, link, flow_current_capacity,
                        unfinished_schedule_flow):
        '''Schedule flows that in link.flows and in unfinished_schedule_flow,
        allocate remain capacities equally to these flows.
        return a dict {flow_obj: flow_schedule_capacity},
        which contains all flow in link.flows, denoting capacities.

        Args:
            link: class Link
            flow_current_capacity: dict, {flow_obj: capacity}, denoting the
                current capacity for each flow
            unfinished_schedule_flow: a set containing all flows who require
                more capacity
        '''
        scheduler_result = {}
        # Find current available capacity
        total_available_capacity = link.capacity
        unfinished_flows_num = 0
        # Calculate remain capacity in link, check how many flows need further
        # calculation
        for flow in link.flows:
            total_available_capacity -= flow_current_capacity[flow]
            if flow not in unfinished_schedule_flow:
                scheduler_result[flow] = flow_current_capacity[flow]
            else:
                unfinished_flows_num += 1
        # Schedule these unfinished flows
        for flow in link.flows:
            if flow in unfinished_schedule_flow:
                scheduler_result[flow] = flow_current_capacity[flow] \
                    + total_available_capacity / unfinished_flows_num
        return scheduler_result
