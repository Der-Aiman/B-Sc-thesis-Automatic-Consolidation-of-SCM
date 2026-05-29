from abc import ABC, abstractmethod
from copy import copy
from math import comb
from numbers import Number
from collections.abc import Callable
from enum import StrEnum
import typing
import numpy as np
import sympy
import random_variable as rv

class ResultType(StrEnum):
    logical = "bool"
    unknown = "unknown"
    float = "float"


class EqElement(ABC):
    result_type: ResultType

    @abstractmethod
    def evaluate(self, values):
        raise NotImplementedError()

    @abstractmethod
    def get_variables(self):
        return set()

    @abstractmethod
    def update(self, vars):
        raise NotImplementedError()

    @abstractmethod
    def replace(self, vars):
        raise NotImplementedError()
    
    @abstractmethod
    def simplify(self):
        if len(self.get_variables()) == 0:
            return Const(self.evaluate({}))
        return self

class Operator(EqElement):

    def get_operands(self):
        raise NotImplementedError()

class FlatOperator(Operator):
    @abstractmethod
    def to_operator_tree(self):
        raise NotImplementedError()

class StructVar(EqElement):
    def __init__(self, _id: str, struct_eq: EqElement|None):
        self.id = _id
        self.struct_eq = struct_eq
        if struct_eq is not None:
            self.result_type = struct_eq.result_type
        else:
            self.result_type = ResultType.unknown

    def evaluate(self, values: dict):
        if (self.id not in values) and (self.struct_eq is not None):
            values[self.id] = self.struct_eq.evaluate(values)
        if self.id not in values:
            raise RuntimeError(f"undefined variable {self.id}")
        return values[self.id]

    def get_variables(self):
        return {self.id}

    def simplify(self):
        return self

    def __str__(self):
        return self.id

    def __eq__(self, other):
        return isinstance(other, StructVar) and (self.id == other.id)

    def update(self, vars):
        if self.id in vars:
            return vars[self.id]
        else:
            return self

    def replace(self, vars):
        return self.struct_eq if self.id in vars else self

class Const(EqElement):
    def __init__(self, value):
        self.value = value
        if isinstance(value, (float, int, rv.ContinuousRV, rv.DiscreteRV)):
            self.result_type = ResultType.float
        elif isinstance(value, (bool, rv.LogicalRV)):
            self.result_type = ResultType.logical
        else:
            self.result_type = ResultType.unknown

    def evaluate(self, _=None):
        return self.value

    def get_variables(self):
        return set()

    def invert(self):
        if isinstance(self.value, bool):
            return Const(not self.value)
        if isinstance(self.value, rv.LogicalRV):
            return Const(~self.value)
        return Const(-1*self.value)

    def simplify(self):
        return self

    def update(self, _):
        return self

    def __str__(self):
        return str(self.value)

    def __eq__(self, other):
        return isinstance(other, Const) and (self.value == other.value)

    def replace(self, _):
        return self

class Sum(Operator):
    def __init__(self, operands):
        self.operands = operands
        self.result_type = ResultType.float

    def evaluate(self, values):
        acc_val = 0
        for op in self.operands:
            acc_val += op.evaluate(values)
        return acc_val

    def get_variables(self):
        vars = set()
        for op in self.operands:
            vars = vars.union(op.get_variables())

        return vars

    def simplify(self):
        new_operands = copy(self.operands)
        to_remove = []
        cond_stmts = []
        cond_counts = []
        for i, op in enumerate(new_operands):
            new_operands[i] = op.simplify()
            if isinstance(new_operands[i], FlatOperator):
                new_operands[i] = new_operands[i]
            if isinstance(new_operands[i], CondSwitch):
                cond_stmts.append(i)
                cond_counts.append(len(new_operands[i].conditions))
        if len(cond_stmts) > 0:
            positions = [0 for _ in range(len(cond_stmts))]
            new_conditions = []
            new_cases = []
            while positions[-1] <= cond_counts[-1]:
                cond_operands = [new_operands[idx].conditions[positions[i]] for i, idx in enumerate(cond_stmts) if positions[i] < len(new_operands[idx].conditions)]
                new_condition = LogicalAnd(cond_operands).simplify()
                new_case_op = copy(new_operands)
                for i, idx in enumerate(cond_stmts):
                    new_case_op[idx] = new_operands[idx].cases[positions[i]]
                new_case = Sum(new_case_op).simplify()
                new_cases.append(new_case)
                if len(cond_operands) == 0:
                    return CondSwitch(new_conditions, new_cases).simplify()
                new_conditions.append(new_condition)
                i = 0
                overflow = True
                while overflow and (i < len(positions)):
                    if positions[i] >= cond_counts[i]:
                        positions[i] = 0
                    else:
                        positions[i] += 1
                        overflow = False
                    i += 1
                if overflow:
                    return CondSwitch(new_conditions, new_cases).simplify()
        for i, op in enumerate(new_operands):
            if isinstance(op, Sum):
                to_remove.append(new_operands[i])
        for op in to_remove:
            new_operands += op.operands
            new_operands.remove(op)
        to_collapse = []
        for op in new_operands:
            if isinstance(op, Const):
                to_collapse.append(op)
        collapsed_val = 0
        for op in to_collapse:
            collapsed_val += op.evaluate({})
            new_operands.remove(op)
        if collapsed_val != 0:
            new_operands.append(Const(collapsed_val))
        if len(new_operands) == 0:
            return Const(0)
        if len(new_operands) == 1:
            return new_operands[0]
        new_node = Sum(new_operands)
        if len(new_node.get_variables()) == 0:
            return Const(new_node.evaluate({}))
        return new_node

    def replace(self, vars=None):
        new_operands = []
        for op in self.operands:
            if isinstance(op, StructVar) and ((vars is None) or (op.id in vars)) and (op.struct_eq is not None):
                new_operands.append(op.struct_eq)
            elif isinstance(op, (StructVar, Const)):
                new_operands.append(op)
            else:
                new_operands.append(op.replace(vars))
        return Sum(new_operands)

    def update(self, vars:dict):
        new_operands = []
        for op in self.operands:
            if isinstance(op, StructVar) and (op.id in vars):
                new_operands.append(vars[op.id])
            else:
                new_operands.append(op.update(vars))
        return Sum(new_operands)

    def __str__(self):
        ret = f"({str(self.operands[0])})"
        for op in self.operands[1:]:
            ret += f" + ({str(op)})"
        return ret

class Product(Operator):
    def __init__(self, operands):
        self.operands = operands
        self.result_type = ResultType.float

    def evaluate(self, values):
        acc_val = 1
        for op in self.operands:
            acc_val *= op.evaluate(values)
        return acc_val

    def get_variables(self):
        vars = set()
        for op in self.operands:
            vars = vars.union(op.get_variables())
        return vars

    def simplify(self):
        new_operands = copy(self.operands)
        cond_stmts = []
        cond_counts = []
        for i, op in enumerate(new_operands):
            new_operands[i] = op.simplify()
            if isinstance(new_operands[i], CondSwitch):
                cond_stmts.append(i)
                cond_counts.append(len(new_operands[i].conditions))
        if len(cond_stmts) > 0:
            positions = [0 for _ in range(len(cond_stmts))]
            new_conditions = []
            new_cases = []
            while positions[-1] <= cond_counts[-1]:
                new_condition = LogicalAnd([new_operands[idx].conditions[positions[i]] for i, idx in enumerate(cond_stmts) if positions[i] < len(new_operands[idx].conditions)]).simplify()
                new_case_op = copy(new_operands)
                for i, idx in enumerate(cond_stmts):
                    new_case_op[idx] = new_operands[idx].cases[positions[i]]
                new_case = Product(new_case_op).simplify()
                new_conditions.append(new_condition)
                new_cases.append(new_case)
                i = 0
                overflow = True
                while overflow and (i < len(positions)):
                    if positions[i] >= cond_counts[i]:
                        positions[i] = 0
                    else:
                        positions[i] += 1
                        overflow = False
                    i += 1
                if overflow:
                    return CondSwitch(new_conditions, new_cases)
        to_remove = []
        for i, op in enumerate(new_operands):
            if isinstance(new_operands[i], Product):
                to_remove.append(new_operands[i])
        for op in to_remove:
            new_operands += op.operands
            new_operands.remove(op)
        to_collapse = []
        for op in new_operands:
            if isinstance(op, Const):
                    to_collapse.append(op)
        collapsed_val = 1
        for op in to_collapse:
            collapsed_val *= op.evaluate({})
            new_operands.remove(op)
        if collapsed_val == 0:
            return Const(0)
        if collapsed_val != 1:
            new_operands.append(Const(collapsed_val))
        if len(new_operands) == 0:
            if len(to_collapse) == 0:
                return Const(0)
            else:
                return Const(1)
        if len(new_operands) == 1:
            return new_operands[0]
        new_node = Product(new_operands)
        if len(new_node.get_variables()) == 0:
            return Const(new_node.evaluate({}))
        return new_node

    def replace(self, vars=None):
        new_operands = []
        for op in self.operands:
            if isinstance(op, StructVar) and ((vars is None) or (op.id in vars)) and (op.struct_eq is not None):
                new_operands.append(op.struct_eq)
            elif isinstance(op, (StructVar, Const)):
                new_operands.append(op)
            else:
                new_operands.append(op.replace(vars))
        return Product(new_operands)

    def update(self, vars:dict):
        new_operands = []
        for op in self.operands:
            if isinstance(op, StructVar) and (op.id in vars):
                new_operands.append(vars[op.id])
            else:
                new_operands.append(op.update(vars))
        return Product(new_operands)

    def __str__(self):
        ret = f"({str(self.operands[0])})"
        for op in self.operands[1:]:
            ret += f" * ({str(op)})"
        return ret

class LogicalNot(Operator):
    def __init__(self, operand: EqElement):
        self.operand = operand
        self.result_type = ResultType.logical

    def evaluate(self, values):
        pre_evaluation = self.operand.evaluate(values)
        if isinstance(pre_evaluation, bool):
            return not self.operand.evaluate(values)
        return ~pre_evaluation
    
    def get_variables(self):
        return self.operand.get_variables()

    def _simplify(self):
        new_operand = self.operand
        if isinstance(new_operand, Operator):
            new_operand = new_operand.simplify()

        if isinstance(new_operand, LogicalNot):
            return new_operand.operand

        if isinstance(new_operand, (LogicalAnd, LogicalOr)):
            return new_operand.invert()

        if isinstance(new_operand, CondSwitch):
            new_cases = []
            for case in new_operand.cases:
                new_cases.append(LogicalNot(case).simplify())
            return CondSwitch(new_operand.conditions, new_cases).simplify()

    def simplify(self):
        if isinstance(self.operand, CondSwitch):
            new_cases = [LogicalNot(op).simplify() for op in self.operand.cases]
            return CondSwitch(copy(self.operand.conditions), new_cases)
        solution = solve(self).simplify()
        if isinstance(solution, Const):
            return solution
        return solution.to_operator_tree()

    def invert(self):
        return self.operand

    def replace(self, vars=None):
        if isinstance(self.operand, Operator):
            return LogicalNot(self.operand.replace(vars))
        elif isinstance(self.operand, StructVar) and ((vars is None) or (self.operand.id in vars)) and (self.operand.struct_eq is not None):
            return LogicalNot(self.operand.struct_eq)
        else:
            return self

    def update(self, vars):
        if isinstance(self.operand, StructVar) and (self.operand.id in vars):
            return LogicalNot(vars[self.operand.id])
        else:
            return LogicalNot(self.operand.update(vars))

    def __str__(self):
        return f"~({str(self.operand)})"

    def __eq__(self, other):
        if not other.evaluates_to() == bool:
            return False
        return solve(self) == solve(other)

class LogicalAnd(Operator):
    def __init__(self, operands):
        self.operands = operands
        self.result_type = ResultType.logical

    def evaluate(self, values):
        acc_val = True
        for op in self.operands:
            acc_val = acc_val & op.evaluate(values)
        return acc_val

    def get_variables(self):
        vars = set()
        for op in self.operands:
            vars = vars.union(op.get_variables())

        return vars

    def simplify(self):
        new_operands = copy(self.operands)
        cond_stmts = []
        cond_counts = []
        for i, op in enumerate(new_operands):
            new_operands[i] = op.simplify()
            if isinstance(new_operands[i], CondSwitch):
                cond_stmts.append(i)
                cond_counts.append(len(new_operands[i].conditions))
        if len(cond_stmts) > 0:
            positions = [0 for _ in range(len(cond_stmts))]
            new_conditions = []
            new_cases = []
            while positions[-1] <= cond_counts[-1]:
                cond_operands = [new_operands[idx].conditions[positions[i]] for i, idx in enumerate(cond_stmts) if positions[i] < len(new_operands[idx].conditions)]
                new_condition = LogicalAnd(cond_operands).simplify()
                new_case_op = copy(new_operands)
                for i, idx in enumerate(cond_stmts):
                    new_case_op[idx] = new_operands[idx].cases[positions[i]]
                new_case = LogicalAnd(new_case_op).simplify()
                new_cases.append(new_case)
                if len(cond_operands) == 0:
                    return CondSwitch(new_conditions, new_cases).simplify()
                new_conditions.append(new_condition)
                i = 0
                overflow = True
                while overflow and (i < len(positions)):
                    if positions[i] >= cond_counts[i]:
                        positions[i] = 0
                    else:
                        positions[i] += 1
                        overflow = False
                    i += 1
                if overflow:
                    return CondSwitch(new_conditions, new_cases).simplify()
        solution = solve(self).simplify()
        if isinstance(solution, Const):
            return solution
        return solution.to_operator_tree()
    
    def update(self, vars:dict):
        new_operands = []
        for op in self.operands:
            if isinstance(op, StructVar) and (op.id in vars):
                new_operands.append(vars[op.id])
            else:
                new_operands.append(op.update(vars))
        return LogicalAnd(new_operands)
    
    def invert(self):
        return LogicalOr([LogicalNot(op) for op in self.operands])
    
    def replace(self, vars=None):
        new_operands = []
        for op in self.operands:
            if isinstance(op, StructVar) and ((vars is None) or (op.id in vars)) and (op.struct_eq is not None):
                new_operands.append(op.struct_eq)
            elif isinstance(op, (StructVar, Const)):
                new_operands.append(op)
            else:
                new_operands.append(op.replace(vars))
        return LogicalAnd(new_operands)

    def __str__(self):
        ret = f"({str(self.operands[0])})"
        for op in self.operands[1:]:
            ret += f" & ({str(op)})"
        return ret

    def __eq__(self, other):
        if not other.evaluates_to() == bool:
            return False
        return solve(self) == solve(other)

class LogicalOr(Operator):
    def __init__(self, operands):
        self.operands = operands
        self.result_type = ResultType.logical

    def evaluate(self, values):
        acc_val = False
        for op in self.operands:
            acc_val = acc_val | op.evaluate(values)
        return acc_val

    def get_variables(self):
        vars = set()
        for op in self.operands:
            vars = vars.union(op.get_variables())

        return vars

    def simplify(self):
        new_operands = copy(self.operands)
        cond_stmts = []
        cond_counts = []
        for i, op in enumerate(new_operands):
            new_operands[i] = op.simplify()
            if isinstance(new_operands[i], CondSwitch):
                cond_stmts.append(i)
                cond_counts.append(len(new_operands[i].conditions))
        if len(cond_stmts) > 0:
            positions = [0 for _ in range(len(cond_stmts))]
            new_conditions = []
            new_cases = []
            while positions[-1] <= cond_counts[-1]:
                cond_operands = [new_operands[idx].conditions[positions[i]] for i, idx in enumerate(cond_stmts) if positions[i] < len(new_operands[idx].conditions)]
                new_condition = LogicalAnd(cond_operands).simplify()
                new_case_op = copy(new_operands)
                for i, idx in enumerate(cond_stmts):
                    new_case_op[idx] = new_operands[idx].cases[positions[i]]
                new_case = LogicalOr(new_case_op).simplify()
                new_cases.append(new_case)
                if len(cond_operands) == 0:
                    return CondSwitch(new_conditions, new_cases).simplify()
                new_conditions.append(new_condition)
                i = 0
                overflow = True
                while overflow and (i < len(positions)):
                    if positions[i] >= cond_counts[i]:
                        positions[i] = 0
                    else:
                        positions[i] += 1
                        overflow = False
                    i += 1
                if overflow:
                    return CondSwitch(new_conditions, new_cases).simplify()
        solution = solve(self).simplify()
        if isinstance(solution, Const):
            return solution
        return solution.to_operator_tree()

    def invert(self):
        return LogicalAnd([LogicalNot(op).simplify() for op in self.operands])

    def replace(self, vars=None):
        new_operands = []
        for op in self.operands:
            if isinstance(op, StructVar) and ((vars is None) or (op.id in vars)) and (op.struct_eq is not None):
                new_operands.append(op.struct_eq)
            elif isinstance(op, (StructVar, Const)):
                new_operands.append(op)
            else:
                new_operands.append(op.replace(vars))
        return LogicalOr(new_operands)

    def update(self, vars:dict):
        new_operands = []
        for op in self.operands:
            if isinstance(op, StructVar) and (op.id in vars):
                new_operands.append(vars[op.id])
            else:
                new_operands.append(op.update(vars))
        return LogicalOr(new_operands)

    def __str__(self):
        ret = f"({str(self.operands[0])})"
        for op in self.operands[1:]:
            ret += f" | ({str(op)})"
        return ret

    def __eq__(self, other):
        if not (other.evaluates_to() == bool):
            return False
        return solve(self) == solve(other)

class Equals(Operator):
    def __init__(self, op1: EqElement, op2: EqElement):
        self.op1 = op1
        self.op2 = op2
        self.result_type = ResultType.logical

    def evaluate(self, values):
        return self.op1.evaluate(values) == self.op2.evaluate(values)

    def get_variables(self):
        return self.op1.get_variables().union(self.op2.get_variables())

    def simplify(self):
        if self.op1 == self.op2:
            return Const(True)
        elif isinstance(self.op1, Const) and isinstance(self.op2, Const):
            return self.op1.value == self.op2.value
        else:
            return self

    def replace(self, vars=None):
        new_op1 = None
        new_op2 = None

        if isinstance(self.op1, Operator):
            new_op1 = self.op1.replace(vars)
        elif isinstance(self.op1, StructVar) and ((vars is None) or (self.op1.id in vars)) and (self.op1.struct_eq is not None):
            new_op1 = self.op1.struct_eq
        else:
            new_op1 = self.op1

        if isinstance(self.op2, Operator):
            new_op2 = self.op2.replace(vars)
        elif isinstance(self.op2, StructVar) and ((vars is None) or (self.op2.id in vars)) and (self.op2.struct_eq is not None):
            new_op2 = self.op2.struct_eq
        else:
            new_op2 = self.op2

        return Equals(new_op1, new_op2)

    def __str__(self):
        return f"({str(self.op1)}) = ({str(self.op2)})"

    def __eq__(self, other):
        return isinstance(other, Equals) and ((self.op1 == other.op1) and (self.op2 == other.op2)) or ((self.op1 == other.op2) and (self.op2 == other.op1))

class Min(Operator):
    def __init__(self, operands):
        self.operands = operands
        self.result_type = ResultType.unknown

    def evaluate(self, values):
        args = [op.evaluate(values) for op in self.operands]
        return min_rv(*args)

    def simplify(self):
        new_operands = []
        for op in self.operands:
            new_op = op.simplify()
            if isinstance(new_op, Min):
                new_operands += new_op.operands
            else:
                new_operands.append(op)
        return Min(new_operands)

    def get_variables(self):
        vars = set()

        for op in self.operands:
            vars = vars.union(op.get_variables())

        return vars

    def update(self, vars):
        new_operands = []
        for op in self.operands:
            if isinstance(op, StructVar) and (op.id in vars):
                new_operands.append(vars[op.id])
            else:
                new_operands.append(op.update(vars))
        return Min(new_operands)

    def replace(self, vars):
        new_operands = []
        for op in self.operands:
            if isinstance(op, StructVar) and ((vars is None) or (op.id in vars)) and (op.struct_eq is not None):
                new_operands.append(op.struct_eq)
            elif isinstance(op, (StructVar, Const)):
                new_operands.append(op)
            else:
                new_operands.append(op.replace(vars))
        return Min(new_operands)
    
    def __str__(self):
        return f"Min{[str(op) for op in self.operands]}"

class CondSwitch(Operator):
    def __init__(self, conditions, cases):
        self.conditions = conditions
        self.cases = cases
        self.result_type = ResultType.unknown
        if len(cases) != len(conditions)+1:
            raise ValueError("Number of cases should be one plus number of conditions")

    def evaluate(self, values):
        for i, cond in enumerate(self.conditions):
            if cond.evaluate(values):
                return self.cases[i].evaluate(values)
        return self.cases[-1].evaluate

    def get_variables(self):
        vars = set()
        for cond in self.conditions:
            vars = vars.union(cond.get_variables())
        for case in self.cases:
            vars = vars.union(case.get_variables())
        return vars

    def replace(self, vars=None):
        new_conditions = []
        new_cases = []
        for cond in self.conditions:
            if isinstance(cond, Operator):
                new_conditions.append(cond.replace(vars))
            elif isinstance(cond, StructVar) and ((vars is None) or (cond.id in vars)):
                new_conditions.append(cond.struct_eq)
            else:
                new_conditions.append(cond)
        for case in self.cases:
            if isinstance(case, Operator):
                new_cases.append(case.replace(vars))
            elif isinstance(case, StructVar) and ((vars is None) or (case.id in vars)):
                new_cases.append(case.struct_eq)
            else:
                new_cases.append(case)

        return CondSwitch(new_conditions, new_cases)

    def simplify(self):
        new_conditions = []
        new_cases = []
        early_break = False
        for i, cond in enumerate(self.conditions):
            if isinstance(self.cases[i], Operator):
                new_case = self.cases[i].simplify()
            else:
                new_case = self.cases[i]
            if not isinstance(new_case, CondSwitch):
                new_cond = cond.simplify()
                if not isinstance(new_cond, Const):
                    new_conditions.append(new_cond)
                    new_cases.append(new_case)
                elif new_cond.value:
                    early_break = True
                    new_cases.append(new_case)
                    break
            else:
                for j, sub_cond in enumerate(new_case.conditions):
                    new_cond = LogicalAnd([cond, sub_cond]).simplify()
                    if not isinstance(new_cond, Const):
                        new_conditions.append(new_cond)
                        new_cases.append(new_case.cases[j])
                    elif new_cond.value:
                        early_break = True
                        new_cases.append(new_case.cases[j])
                        break
                if early_break:
                    break
                else:
                    new_cond = cond.simplify()
                    if not isinstance(new_cond, Const):
                        new_conditions.append(new_cond)
                        new_cases.append(new_case.cases[-1])
                    elif new_cond.value:
                        early_break = True
                        new_cases.append(new_case.cases[-1])
                        break
        if not early_break:
            new_case = self.cases[-1].simplify()
            if not isinstance(new_case, CondSwitch):
                new_cases.append(new_case)
            else:
                for i, cond in enumerate(new_case.conditions):
                    new_cond = cond.simplify()
                    if not isinstance(new_cond, Const):
                        new_conditions.append(new_cond)
                        new_cases.append(new_case.cases[i])
                    elif new_cond.value:
                        new_cases.append(new_case.cases[i])
                        break
                if len(new_conditions) == len(new_cases):
                    new_cases.append(new_case.cases[-1])
        if len(new_conditions) == 0:
            return new_cases[0]
        return CondSwitch(new_conditions, new_cases)

    def update(self, vars:dict):
        new_conditions = []
        for op in self.conditions:
            if isinstance(op, StructVar) and (op.id in vars):
                new_conditions.append(vars[op.id])
            else:
                new_conditions.append(op.update(vars))

        new_cases = []
        for op in self.cases:
            if isinstance(op, StructVar) and (op.id in vars):
                new_cases.append(vars[op.id])
            else:
                new_cases.append(op.update(vars))
        return CondSwitch(new_conditions, new_cases)

class KDNFSolution(FlatOperator):
    def __init__(self, variables: dict[str, StructVar], positions: dict[str, int], solutions: list[str]):
        self.variables = variables
        self.solutions = solutions
        self.positions = positions
        self.sorted_vars = sorted(positions.keys(), key=lambda v: positions[v])
        self.result_type = ResultType.logical

    def invert(self):
        new_solutions = sorted(set(power_sol(len(self.variables))).difference(self.solutions))
        return KDNFSolution(self.variables, self.positions, new_solutions)

    def simplify(self):
        if ((len(self.solutions) == 2**len(self.variables))) or ((len(self.solutions) == 1) and (self.solutions[0] == "*"*len(self.solutions[0]))):
            return Const(True)
        if len(self.solutions) == 0:
            return Const(False)
        classes = [[] for _ in range(len(self.variables)+1)]
        sub_solutions = {}
        for sol in self.solutions:
            classes[sol.count("1")].append(sol)
        for c1, sols in enumerate(classes[:-1]):
            for sol1 in sols:
                for sol2 in classes[c1+1]:
                    eq_map = str_xor(sol1, sol2)
                    if eq_map.count("0") > 1:
                        continue
                    neq = eq_map.index("0")
                    var_idx = neq - sol1[:neq].count("*")
                    if neq not in sub_solutions:
                        sub_solutions[neq] = KDNFSolution({var:self.variables[var] for var in self.variables.keys() if var != self.sorted_vars[var_idx]}, {var:self.positions[var] for var in self.variables.keys() if var != self.sorted_vars[var_idx]}, [])
                    new_sol = sol1[:neq]+ "*" + sol1[neq+1:]
                    if new_sol not in sub_solutions[neq].solutions:
                        sub_solutions[neq].solutions.append(new_sol)
        to_merge = [self]
        for sub_sol in sub_solutions.values():
            to_merge.append(sub_sol.simplify())
        return merge_solutions(to_merge)

    def reorder(self, positions: list[str]|dict[str,int]):
        if isinstance(positions, dict):
            positions = sorted(positions.keys(), key=lambda x: self.positions[x])
        new_solutions = []
        for sol in self.solutions:
            reorder = ""
            for var in positions:
                reorder += sol[self.positions[var]]
            new_solutions.append(reorder)
        return KDNFSolution(self.variables, {v:i for i, v in enumerate(positions)}, new_solutions)

    def evaluate(self, values):
        var_vals = ""
        for var in self.sorted_vars:
            if self.variables[var].evaluate(values):
                var_vals += "1"
            else:
                var_vals += "0"
        for sol in self.solutions:
            eq = True
            for i, char in enumerate(var_vals):
                if (char != sol[i]) and (sol[i] != "*"):
                    eq = False
                    break
            if eq:
                return True
        return False

    def evaluate_rv(self, values: dict[str, rv.LogicalRV|bool]):
        if len(set(self.variables.keys()).difference(values.keys())) > 0:
            raise RuntimeError("all values must be set for rv evaluation.")
        final_prob = 0
        for t in self.solutions:
            term_prob = 1
            i = 0
            for index, c in enumerate(t):
                if c == "1":
                    if isinstance(values[self.sorted_vars[i]], rv.LogicalRV):
                        term_prob *= values[self.sorted_vars[i]].true_prob
                        i += 1
                    elif isinstance(values[self.sorted_vars[i]], bool) and not values[self.sorted_vars[i]]:
                        term_prob = 0
                        i += 1
                elif c == "0":
                    if isinstance(values[self.sorted_vars[i]], rv.LogicalRV):
                        term_prob *= 1 - values[self.sorted_vars[i]].true_prob
                        i += 1
                    elif isinstance(values[self.sorted_vars[i]], bool) and values[self.sorted_vars[i]]:
                        term_prob = 0
                        i += 1
                else:
                    if self.positions[self.sorted_vars[i]] == index:
                        i += 1
                if term_prob == 1:
                    return True
                final_prob += term_prob
            if final_prob == 0:
                return False
            if final_prob == 1:
                return True
            return rv.LogicalRV(final_prob)

    def replace(self, vars):
        ot = self.to_operator_tree()
        if isinstance(ot, Const):
            return ot
        return solve(self.to_operator_tree().replace(vars))

    def get_variables(self):
        return set(self.variables.keys())

    def to_operator_tree(self):
        conjunctions = []
        for sol in self.solutions:
            term = []
            for var_id, var in self.variables.items():
                if sol[self.positions[var_id]] == "1":
                    term.append(var)
                elif sol[self.positions[var_id]] == "0":
                    term.append(LogicalNot(var))
            if len(term) > 1:
                term = LogicalAnd(term)
            elif len(term) == 1:
                term = term[0]
            else:
                return Const(True)
            conjunctions.append(term)
        if len(conjunctions) == 1:
            return conjunctions[0]
        return LogicalOr(conjunctions)

    def remove_var(self, var):
        new_solutions = []
        for sol in self.solutions:
            new_solutions.append(sol[:self.positions[var]]+sol[self.positions[var]+1:])
        new_vars = {var_id:variable for var_id, variable in self.variables.items() if var_id != var}
        new_positions = {var_id:pos if pos < self.positions[var] else pos-1 for var_id, pos in self.positions.items() if var_id != var}
        return KDNFSolution(new_vars, new_positions, new_solutions)

    def add_solution(self, sol):
        if sol not in self.solutions:
            self.solutions.append(sol)

    def remove_solution(self, sol):
        if sol in self.solutions:
            del self.solutions[self.solutions.index(sol)]

    def update(self, vars):
        return solve(self.to_operator_tree().update(vars))

    def get_exactly_structures(self):
        buckets = {}
        exactlys = []
        new_sols = copy(self.solutions)
        for sol in self.solutions:
            count = sol.count("1")
            if count not in buckets:
                buckets[count] = []
            buckets[count].append(sol)
        for count, sols in buckets:
            if len(sols) == comb(len(self.variables), count):
                for sol in sols:
                    del new_sols[new_sols.index(sol)]
                exactlys.append(ExactlyStructure(copy(self.variables), count))
        if len(exactlys) == 0:
            return self
        if len(new_sols) > 0:
            exactlys.append(KDNFSolution(copy(self.variables), copy(self.positions), new_sols))
        if len(exactlys) == 1:
            return exactlys[0]
        return LogicalOr(exactlys)

    def get_exactlys_thorough(self):
        simple_solution = self.simplify()
        subsets = {}
        for sol in simple_solution.solutions:
            subset_idx = 0
            for i, entry in enumerate(sol):
                if entry == "1":
                    subset_idx += 2**i
            if subset_idx not in subsets:
                subsets[subset_idx] = []
            subsets[subset_idx].append(sol)
        sub_solutions = []
        for idx, sols in subsets:
            variables = {}
            for i, entry in enumerate(reversed(bin(idx))):
                if entry == "b":
                    break
                if entry == "1":
                    variables[self.sorted_vars[i]] = self.variables[self.sorted_vars[i]]
            sub_solutions.append(KDNFSolution(variables, self.positions, sols))
        exactly_structures = [sub_sol.get_exactly_structures() for sub_sol in sub_solutions]
        to_merge = []
        operands = []
        for o in exactly_structures:
            if isinstance(o, KDNFSolution):
                to_merge.append(o)
            elif isinstance(o, ExactlyStructure):
                operands.append(o)
            elif isinstance(o, LogicalOr):
                operands += o.operands[:-1]
                if isinstance(o.operands[-1], KDNFSolution):
                    to_merge.append(o.operands[-1])
                elif isinstance(o.operands[-1], ExactlyStructure):
                    operands.append(o.operands[-1])
        operands.append(merge_solutions(to_merge))
        return LogicalOr(operands)

    def __mul__(self, other):
        if not isinstance(other, (KDNFSolution, Const)):
            raise TypeError("Both operands should be of type KDNFSolution or Const")
        if isinstance(other, Const):
            if other.value:
                return self
            else:
                return Const(False)
        intersection = set(self.variables.keys()).intersection(other.variables.keys())
        to_add = list(set(other.variables.keys()).difference(intersection))
        new_var_set = set(self.variables.keys()).union(other.variables.keys())
        new_vars = {key:self.variables[key] if key in self.variables else other.variables[key] for key in new_var_set}
        new_solutions = []
        new_positions = copy(self.positions)
        for i, var in enumerate(to_add):
            new_positions[var] = len(self.variables)+i
        for base_sol in self.solutions:
            for add_sol in other.solutions:
                skip = False in [base_sol[self.positions[v]] == add_sol[other.positions[v]] for v in intersection]
                if not skip:
                    new_sol = base_sol
                    for var in to_add:
                        new_sol += add_sol[other.positions[var]]
                    new_solutions.append(new_sol)
        return KDNFSolution(new_vars, new_positions, new_solutions)

    def __add__(self, other):
        if not isinstance(other, (KDNFSolution, Const)):
            raise TypeError("Both operands should be of type KDNFSolution or Const")
        if isinstance(other, Const):
            if other.value:
                return Const(True)
            return self
        intersection = set(self.variables.keys()).intersection(other.variables.keys())
        to_add = list(set(other.variables.keys()).difference(intersection))
        self_add_values = power_sol(len(to_add))
        other_add_values = power_sol(len(set(self.variables).difference(intersection)))
        if len(self_add_values) == 0:
            self_add_values = [""]
        if len(other_add_values) == 0:
            other_add_values = [""]
        new_var_set = set(self.variables.keys()).union(other.variables.keys())
        new_vars = {key:self.variables[key] if key in self.variables else other.variables[key] for key in new_var_set}
        new_solutions = []
        new_positions = copy(self.positions)
        for i, var in enumerate(to_add):
            new_positions[var] = len(self.variables)+i
        for sol in self.solutions:
            for subterm in self_add_values:
                new_solutions.append(sol+subterm)
        for sol in other.solutions:
            for subterm in other_add_values:
                new_sol = ""
                i=0
                for var in sorted(new_vars.keys(), key = lambda v: new_positions[v]):
                    if var in other.variables:
                        new_sol += sol[other.positions[var]]
                    elif var in self.variables:
                        new_sol += subterm[i]
                        i+=1
                if new_sol not in new_solutions:
                    new_solutions.append(new_sol)

        return KDNFSolution(new_vars, new_positions, new_solutions)

    def __eq__(self, other):
        if isinstance(other, (LogicalAnd, LogicalOr, LogicalNot)):
            other = solve(other)
        for sv, ov in zip(sorted(self.variables.keys()), sorted(other.variables.keys())):
            if sv != ov:
                return False
        for ss, os in zip(sorted(self.solutions), sorted(other.reorder(self.positions).solutions)):
            if ss != os:
                return False
        return True

    def get_symbol(self):
        raise RuntimeError("KDNFSolution does not have a symbol")

def power_sol(length) -> list[str]:
    if length == 0:
        return []
    ret = []
    for i in range(2**length):
        binary = bin(i)[2:]
        ret.append((length-len(binary))*"0" + binary)
    return ret

def solve(logical_operator):
    if isinstance(logical_operator, Const):
        return logical_operator
    if isinstance(logical_operator, StructVar):
        return KDNFSolution({logical_operator.id:logical_operator}, {logical_operator.id:0}, ["1"])
    if isinstance(logical_operator, LogicalNot):
        return solve(logical_operator.operand).invert()
    if isinstance(logical_operator, LogicalAnd):
        solutions = []
        for op in logical_operator.operands:
            sol = solve(op)
            if isinstance(sol, Const) and (not sol.evaluate()):
                return Const(False)
            if isinstance(sol, KDNFSolution):
                solutions.append(sol)
        if len(solutions) == 0:
            return Const(True)
        new_solution = solutions[0]
        for sol in solutions[1:]:
            new_solution *= sol
        return new_solution
    if isinstance(logical_operator, LogicalOr):
        solutions = []
        for op in logical_operator.operands:
            sol = solve(op)
            if isinstance(sol, Const) and sol.evaluate():
                return Const(True)
            if isinstance(sol, KDNFSolution):
                solutions.append(sol)
        if len(solutions) == 0:
            return Const(False)
        new_solution = solutions[0]
        for sol in solutions[1:]:
            new_solution += sol
        return new_solution
    raise RuntimeError(f"Something went wrong while trying to solve {logical_operator}")

class LinearFunction(FlatOperator):
    def __init__(self, variables: dict[str, StructVar], params: list[float], positions: dict[str, int]):
        self.variables = variables
        self.params = params
        self.positions = positions
        self.result_type = ResultType.float

    def get_param(self, var_idx):
        if var_idx is None:
            return self.params[-1]
        return self.params[self.positions[var_idx]]

    def evaluate(self, values):
        acc_val = 0
        for idx, var in self.variables.items():
            acc_val += self.get_param(idx) * var.evaluate(values)
        return acc_val + self.params[-1]

    def replace(self, vars):
        to_remove = {}
        new_params = copy(self.params)
        new_variables = copy(self.variables)
        new_positions = copy(self.positions)
        reverse_vars = sorted(self.variables.keys(), key=lambda v: self.positions[v], reverse=True)
        for var in reverse_vars:
            if var in vars:
                to_remove[var] = self.variables[var]
        shift = len(to_remove)
        remove_params = {idx:self.get_param(idx) for idx in to_remove.keys()}
        for var in reverse_vars:
            if var not in to_remove.keys():
                new_positions[var] -= shift
            else:
                del new_params[new_positions[var]]
                del new_positions[var]
                del new_variables[var]
                shift -= 1

        new_fun = LinearFunction(new_variables, new_params, new_positions)
        cond_stmt = []
        cond_counts = []
        cond_vars = []
        for var in to_remove.values():
            if isinstance(var.struct_eq, CondSwitch):
                cond_vars.append(var)
                cond_stmt.append(var.struct_eq)
                cond_counts.append(len(var.struct_eq.conditions))

        if len(cond_stmt) > 0:
            positions = [0 for _ in range(len(cond_stmt))]
            new_conditions = []
            new_cases = []
            while positions[-1] <= cond_counts[-1]:
                new_condition = LogicalAnd([stmt.conditions[positions[i]] for i, stmt in enumerate(cond_stmt) if positions[i] < len(stmt.conditions)]).simplify()
                new_case = copy(new_fun)
                for i, stmt in cond_stmt:
                    case_comp = stmt.cases[positions[i]]
                    if isinstance(case_comp, LinearFunction) and isinstance(new_case, LinearFunction):
                        new_case.merge(case_comp, remove_params[cond_vars[i].id])
                    elif isinstance(case_comp, LinearFunction) and not isinstance(new_case, LinearFunction):
                        new_case.operands[0].merge(case_comp, remove_params[cond_vars[i].id])
                    elif not isinstance(case_comp, LinearFunction) and isinstance(new_case, LinearFunction):
                        new_case = Sum([new_case, Product([Const(remove_params[cond_vars[i].id]), stmt])])
                    else:
                        new_case.operands.append(stmt)
                new_conditions.append(new_condition)
                new_cases.append(new_case)
                i = 0
                overflow = True
                while overflow and (i < len(positions)):
                    if positions[i] >= cond_counts[i]:
                        positions[i] = 0
                    else:
                        positions[i] += 1
                        overflow = False
                    i += 1
                if overflow:
                    new_fun = CondSwitch(new_conditions, new_cases)
                    break

        for var in to_remove.values():
            if var in cond_vars:
                continue
            if not isinstance(new_fun, CondSwitch):
                if isinstance(var.struct_eq, LinearFunction) and isinstance(new_fun, LinearFunction):
                    new_fun.merge(var.struct_eq, remove_params[var.id])
                elif isinstance(var.struct_eq, LinearFunction) and not isinstance(new_fun, LinearFunction):
                    new_fun.operands[0].merge(var.struct_eq, remove_params[var.id])
                elif not isinstance(var.struct_eq, LinearFunction) and isinstance(new_fun, LinearFunction):
                    new_fun = Sum([new_fun, Product([Const(remove_params[var.id]), var.struct_eq])])
                else:
                    new_fun.operands.append(var.struct_eq)
            else:
                for i, case in enumerate(new_fun.cases):
                    if isinstance(var.struct_eq, LinearFunction) and isinstance(case, LinearFunction):
                        new_fun.cases[i].merge(var.struct_eq, remove_params[var.id])
                    elif isinstance(var.struct_eq, LinearFunction) and not isinstance(case, LinearFunction):
                        new_fun.cases[i].operands[0].merge(var.struct_eq, remove_params[var.id])
                    elif not isinstance(var.struct_eq, LinearFunction) and isinstance(case, LinearFunction):
                        new_fun.cases[i] = Sum([new_fun, Product([Const(remove_params[var.id]), var.struct_eq])])
                    else:
                        new_fun.cases[i].operands.append(var.struct_eq)
        return new_fun

    def get_variables(self):
        return set(self.variables.keys())

    def update(self, vars):
        to_remove = {}
        new_params = copy(self.params)
        new_variables = copy(self.variables)
        new_positions = copy(self.positions)
        reverse_vars = sorted(self.variables.keys(), key=lambda v: self.positions[v], reverse=True)
        for var in reverse_vars:
            if var in vars:
                to_remove[var] = new_variables[var]
        shift = len(to_remove)
        remove_params = {idx:self.get_param(idx) for idx in to_remove.keys()}
        for var in reverse_vars:
            if var not in to_remove.keys():
                new_positions[var] -= shift
            else:
                del new_params[new_positions[var]]
                del new_positions[var]
                del new_variables[var]
                shift -= 1
        new_fun = LinearFunction(new_variables, new_params, new_positions)

        cond_stmt = []
        cond_counts = []
        cond_vars = []
        for var in to_remove.values():
            if isinstance(vars[var.id], CondSwitch):
                cond_vars.append(var.id)
                cond_stmt.append(var.struct_eq)
                cond_counts.append(len(var.struct_eq.conditions))

        if len(cond_stmt) > 0:
            positions = [0 for _ in range(len(cond_stmt))]
            new_conditions = []
            new_cases = []
            while positions[-1] <= cond_counts[-1]:
                new_condition = LogicalAnd([stmt.conditions[positions[i]] for i, stmt in enumerate(cond_stmt) if positions[i] < len(stmt.conditions)]).simplify()
                new_case = copy(new_fun)
                for i, stmt in cond_stmt:
                    case_comp = stmt.cases[positions[i]]
                    if isinstance(case_comp, LinearFunction) and isinstance(new_case, LinearFunction):
                        new_case.merge(case_comp, remove_params[cond_vars[i].id])
                    elif isinstance(case_comp, LinearFunction) and not isinstance(new_case, LinearFunction):
                        new_case.operands[0].merge(case_comp, remove_params[cond_vars[i].id])
                    elif not isinstance(case_comp, LinearFunction) and isinstance(new_case, LinearFunction):
                        new_case = Sum([new_case, Product([Const(remove_params[cond_vars[i].id]), stmt])])
                    else:
                        new_case.operands.append(stmt)
                new_conditions.append(new_condition)
                new_cases.append(new_case)
                i = 0
                overflow = True
                while overflow and (i < len(positions)):
                    if positions[i] >= cond_counts[i]:
                        positions[i] = 0
                    else:
                        positions[i] += 1
                        overflow = False
                    i += 1
                if overflow:
                    new_fun = CondSwitch(new_conditions, new_cases)
                    break

        for var in sorted(self.variables.keys(), key=lambda v: self.positions[v]):
            if var in cond_vars:
                continue
            if var in vars.keys():
                if not isinstance(new_fun, CondSwitch):
                    if isinstance(vars[var], LinearFunction) and isinstance(new_fun, LinearFunction):
                        new_fun.merge(vars[var], remove_params[var])
                    elif isinstance(vars[var], LinearFunction) and not isinstance(new_fun, LinearFunction):
                        new_fun.operands[0].merge(vars[var], remove_params[var])
                    elif not isinstance(vars[var], LinearFunction) and isinstance(new_fun, LinearFunction):
                        new_fun = Sum([new_fun, Product([Const(remove_params[var]), vars[var]])])
                    else:
                        new_fun.operands.append(vars[var])
                else:
                    for i, case in enumerate(new_fun.cases):
                        if isinstance(vars[var], LinearFunction) and isinstance(case, LinearFunction):
                            new_fun.cases[i].merge(vars[var], remove_params[var.id])
                        elif isinstance(vars[var], LinearFunction) and not isinstance(case, LinearFunction):
                            new_fun.cases[i].operands[0].merge(vars[var], remove_params[var.id])
                        elif not isinstance(vars[var], LinearFunction) and isinstance(case, LinearFunction):
                            new_fun.cases[i] = Sum([new_fun, Product([Const(remove_params[var.id]), vars[var]])])
                        else:
                            new_fun.cases[i].operands.append(vars[var])

        return new_fun

    def simplify(self):
        new_variables = {}
        new_params = []
        new_positions = {}
        position = 0
        for var_id in self.variables.keys():
            if self.params[self.positions[var_id]] != 0:
                new_variables[var_id] = self.variables[var_id]
                new_params.append(self.params[self.positions[var_id]])
                new_positions[var_id] = position
            else:
                position += 1
        new_params.append(self.params[-1])
        return LinearFunction(new_variables, new_params, new_positions)

    def merge(self, other, factor: float =1):
        for var in other.variables.keys():
            if var in self.variables:
                self.params[self.positions[var]] += factor * other.get_param(var)
            else:
                self.positions[var] = len(self.variables)
                self.variables[var] = other.variables[var]
                self.params.insert(len(self.params)-1, other.get_param(var))
        self.params[-1] += factor * other.params[-1]

    def substitute(self, var):
        new_params = copy(self.params)
        new_variables = copy(self.variables)
        new_positions = copy(self.positions)

        fac = self.get_param(var)
        subs = self.variables[var].struct_eq
        for v_id in sorted(self.variables.keys(), key=lambda v: self.positions[v])[self.positions[var]:]:
            new_positions[v_id] -= 1

        del new_variables[var]
        del new_params[self.positions[var]]
        del new_positions[var]

        new_fun = LinearFunction(new_variables, new_params, new_positions)

        if isinstance(subs, LinearFunction):
            return new_fun.merge(subs, fac)
        return Sum([new_fun, Product([Const(fac), subs])])

    def to_operator_tree(self):
        return Sum([Product([self.get_param(v_id), v]) for v_id, v in self.variables.items()] + [Const(self.get_param(None))])

    def __str__(self):
        return str(self.to_operator_tree())

def merge_solutions(solutions, solution2=None):
    if not ((isinstance(solutions, (list, tuple)) and (solution2 is None)) or (isinstance(solutions, (KDNFSolution, Const)) and isinstance(solution2, (KDNFSolution, Const)))):
        raise ValueError()
    if isinstance(solutions, (list, tuple)):
        acc = solutions[0]
        for sol in solutions[1:]:
            acc = merge_solutions(acc, sol)
        return acc
    else:
        if isinstance(solutions, Const) and (not solutions.value):
            return solution2
        if isinstance(solution2, Const) and (not solution2.value):
            return solutions
        if (isinstance(solutions, Const) and solutions.value) or (isinstance(solution2, Const) and solution2.value):
            return Const(True)
        new_variables = solutions.variables
        new_variables.update(solution2.variables)
        new_positions = solutions.positions
        new_positions.update(solution2.positions)
        new_solutions = copy(solutions.solutions)
        for to_add in solution2.solutions:
            if to_add == "*"*len(to_add):
                return Const(True)
            add =True
            to_remove = []
            for comp in new_solutions:
                if match_assignment(to_add, comp):
                    add = False
                    break
                if match_assignment(comp, to_add):
                    to_remove.append(comp)
            for item in to_remove:
                new_solutions.remove(item)
            if add:
                new_solutions.append(to_add)
        if len(new_solutions) == 0:
            return Const(False)
        if len(new_solutions) == 2**len(new_variables):
            return Const(True)
        return KDNFSolution(new_variables, new_positions, new_solutions)

def str_xor(str1, str2):
    if not (isinstance(str1, str) and isinstance(str2, str) and (len(str1) == len(str2))):
        raise ValueError("Inputs must be strings of equal length")
    ret = ""
    for i in range(len(str1)):
        ret += "1" if str1[i] == str2[i] else "0"
    return ret

def match_assignment(assignment, solution):
    for i, char in enumerate(assignment):
        if (solution[i] != "*") and (char != solution[i]):
            return False
    return True

def min_rv(*args):
    for a in args:
        if isinstance(a, rv.RandomVariable):
            return rv.rv_min(*args)
    return min(*args)

CONST_TRUE = Const(True)
CONST_FALSE = Const(False)