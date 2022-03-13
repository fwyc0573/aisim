# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import copy
from superscaler.plan_gen.plan.plan import Plan
from superscaler.plan_gen.plan.node_list import NodeList


class AllreducePlan(Plan):
    """ An class that generates a optimized plan from nodelist.
        Targeting for nodes with Allreduce op.
    """

    def __init__(self, plan_name):
        ''' Init a plan with name and set plan_type as Allreduce internally
        Args:
            name: string, e.g. Allreduce_Plan
            type: string, e.g. Allreduce
        '''
        super().__init__(plan_type="Allreduce",
                         plan_name=plan_name)

    def generate_plan(self):
        '''
        Generating plan includes three step:
        1. Find all allreudce nodes as allreduce_node_list
        2. For a specific node, find its related nodes as endpoints
        3. Separate all allreduce node to ring allreduce nodes
        '''

        # Check input node list
        if not isinstance(self._get_node_list(), NodeList):
            return None

        # record original node_list for plan generator
        node_list_ref = self._get_node_list()
        output_node_list = NodeList()

        allreduce_node_list, endpoints = self.find_all_allreduce_nodes(
            node_list_ref, output_node_list)

        index = 0
        for node in allreduce_node_list:
            if node.op == self.get_plan_type():
                key = node.op
                if isinstance(node.name, str):
                    key += node.name
                if isinstance(node.tensor_name, str):
                    key += node.tensor_name
                if node.tensor_name is None:
                    key += "None"
                endpoint = endpoints[key]
                self.separate_allreduce_node(node, endpoint, output_node_list)
        self.set_node_list(output_node_list)

        return self._get_node_list()

    def find_all_allreduce_nodes(self, node_list, output_node_list):
        ''' Return a allreduce_node_list with allreduce op
            Return a endpoints where all nodes have same op, op name
        and tensor_name
        Args:
            node_list: list, the input nodelist
            output_node_list: list, the output nodelist
        '''
        allreduce_node_list = NodeList()
        endpoints = {}
        for node in node_list:
            if node.op == self.get_plan_type():
                allreduce_node_list.append(node)

                key = node.op
                if isinstance(node.name, str):
                    key += node.name
                if isinstance(node.tensor_name, str):
                    key += node.tensor_name
                if node.tensor_name is None:
                    key += "None"
                if key not in endpoints:
                    endpoints[key] = NodeList()
                endpoints[key].append(node)
            else:
                output_node_list.append(node)
        return allreduce_node_list, endpoints

    def separate_allreduce_node(self, node, endpoint, output_node_list):
        '''
        Virtual funtion

        Args:
            node: dict, the node with allreduce op
            endpoint: list, all node enrolled in the same allreduce operator
        '''
        return
