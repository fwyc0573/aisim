# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


class ParserError(Exception):
    """Exception raised for errors in the input.

    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """

    def __init__(self, expression):
        self.expression = expression


class DAGParser():
    def __init__(self, parser_type):
        self.type = parser_type

    def get_parser_type(self):
        return self.type
