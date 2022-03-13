# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from superscaler.plan_gen.plan.allreduce_plan import AllreducePlan


class RawAllreducePlan(AllreducePlan):

    def __init__(self, plan_name="Raw_allreduce_plan"):
        super().__init__(plan_name=plan_name)

    def separate_allreduce_node(self, node, endpoint, output_node_list):
        output_node_list.append(node)

