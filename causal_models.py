import sys
import sympy
import numpy as np
from abc import ABC, abstractmethod
from copy import copy
from itertools import chain, combinations
import random_variable as rv

sys.path.append(".")

import operators

class CausalGraph:
    def __init__(self, ex_vars: list[str], en_vars: list[str], edges: list[tuple[str, str]]):
        self.ex_vars = ex_vars
        self.en_vars = en_vars
        self.edges = edges

    def get_children(self, var):
        children = set()
        for e in self.edges:
            if e[0] == var:
                children.add(e[1])
        return children

    def get_parents(self, var):
        pars = set()
        for e in self.edges:
            if e[1] == var:
                pars.add(e[0])
        return pars

    def get_descendants(self, var):
        descs = set()
        children = self.get_children(var)
        for c in children:
            if c not in descs:
                descs = descs.union(self.get_descendants(c))
        descs = descs.union(children)
        return descs

    def get_predecessors(self, var):
        preds = set()
        pars = self.get_parents(var)
        for p in pars:
            if p not in preds:
                preds = preds.union(self.get_predecessors(p))
        preds = preds.union(pars)
        return preds

    def get_set_parents(self, vars):
        pars = set()
        for v in vars:
            pars = pars.union(self.get_parents(v))
        return pars

    def get_set_children(self, vars):
        children = set()
        for v in vars:
            children = children.union(self.get_children(v))
        return children

    def get_set_predecessors(self, vars):
        preds = set()
        for v in vars:
            if v not in preds:
                preds = preds.union(self.get_predecessors(v))
        return preds

    def get_set_descendants(self, vars):
        descs = set()
        for v in vars:
            if v not in descs:
                descs = descs.union(self.get_descendants(v))
        return descs

    def remove(self, vars):
        new_ex_vars = copy(self.ex_vars)
        new_en_vars = copy(self.en_vars)
        new_edges = copy(self.edges)
        to_remove = []
        for v in vars:
            if v in new_ex_vars:
                new_ex_vars.remove(v)
            if v in new_en_vars:
                new_en_vars.remove(v)
        for e in new_edges:
            if (e[0] not in new_ex_vars and e[0] not in new_en_vars) or e[1] not in new_en_vars:
                to_remove.append(e)
        for e in to_remove:
            new_edges.remove(e)
        return CausalGraph(new_ex_vars, new_en_vars, new_edges)

    def add_en_vars(self, vars):
        for v in vars:
            if v not in self.en_vars:
                self.en_vars.append(v)

    def add_ex_vars(self, vars):
        for v in vars:
            if v not in self.ex_vars:
                self.ex_vars.append(v)

    def add_edges(self, edges):
        for add_e in edges:
            coll = False
            for e in self.edges:
                if e[0] == add_e[0] and e[1] == add_e[1]:
                    coll = True
                    break
            if not coll:
                self.edges.append(add_e)

    def get_var_ordering(self):
        order_map = {v:0 for v in self.ex_vars}
        to_process = copy(self.ex_vars)
        while len(to_process) > 0:
            children = self.get_children(to_process[0])
            to_process += list(children)
            for c in children:
                if order_map.get(c, 0) <= order_map[to_process[0]]:
                    order_map[c] = order_map[to_process[0]]+1
            to_process.remove(to_process[0])
        return sorted(self.en_vars, key=lambda v:order_map[v])

    def focus(self, nodes):
        rel_ex_nodes = [ex_node for ex_node in self.ex_vars if ex_node in nodes]
        rel_en_nodes = [en_node for en_node in self.en_vars if en_node in nodes]
        rel_edges = [edge for edge in self.edges if edge[0] in nodes and edge[1] in nodes]
        return CausalGraph(rel_ex_nodes, rel_en_nodes, rel_edges)

    def recurse_blocked(self, B: list[str], t: str, remaining: set[str], processed: set):
        remaining = copy(remaining)
        processed = copy(processed)
        if t in processed:
            processed.add(t)
            return remaining, processed
        for v in self.get_parents(t):
            if v not in B:
                if v in remaining:
                    remaining.remove(v)
                processed.add(v)
                remaining, processed = self.recurse_blocked(B, v, remaining, processed)
        return remaining, processed

    def compute_blocked(self, B: list[str], T: list[str]|str):
        if isinstance(T, str):
            T = [T]
        remaining = self.get_set_predecessors(B).intersection(self.get_set_predecessors(T))
        processed = set()
        for t in T:
            remaining, processed = self.recurse_blocked(B, t, remaining, processed)
        return remaining

def part_causal_graph_foreward(graph, border_vars):
    if len(graph.en_vars) == 0:
        return []
    running_borders = copy(border_vars)
    border_preds = set()
    node_sets = []
    for border_var in border_vars:
        if border_var not in border_preds:
            border_preds = border_preds.union(graph.get_predecessors(border_var))
    node_sets.append(list(set(graph.en_vars).difference(border_preds)))
    for border in border_vars:
        if border in node_sets[0]:
            running_borders.remove(border)
    node_sets = part_causal_graph_foreward(graph.remove(node_sets[0]), running_borders) + node_sets
    return node_sets

def part_causal_graph_backward(graph, border_vars):
    if len(graph.en_vars) == 0:
        return []
    running_borders = copy(border_vars)
    border_desc = set()
    node_sets = []
    for border_var in border_vars:
        if border_var not in border_desc:
            border_desc = border_desc.union(graph.get_descendants(border_var))
    node_sets.append(list(set(graph.en_vars).difference(border_desc)))
    for border in border_vars:
        if border in node_sets[0]:
            running_borders.remove(border)
    node_sets = part_causal_graph_backward(graph.remove(node_sets[0]), running_borders) + node_sets
    return node_sets

class AbstractSCM(ABC):
    @abstractmethod
    def evaluate_var(self, var, values):
        raise NotImplementedError()

    @abstractmethod
    def evaluate_vars(self, vars, values) -> dict:
        raise NotImplementedError()

    @abstractmethod
    def get_causal_graph(self) -> CausalGraph:
        raise NotImplementedError()

    @abstractmethod
    def marginalize(self, vars=None):
        raise NotImplementedError()

    @abstractmethod
    def marginalize_except(self, inter_vars):
        raise NotImplementedError()

    @abstractmethod
    def get_variables(self) -> set[str]:
        raise NotImplementedError()

    @abstractmethod
    def get_ex_variables(self) -> set[str]:
        raise NotImplementedError()

    @abstractmethod
    def get_en_variables(self) -> set[str]:
        raise NotImplementedError()

    @abstractmethod
    def get_parents(self, var):
        raise NotImplementedError()

    def simulate_interventions(self, intervention_vars=None):
        if intervention_vars is None:
            allowed_interventions = powerset(self.get_en_variables())
        else:
            allowed_interventions = powerset(intervention_vars)
        inter_scm = []
        for s in allowed_interventions:
            inter_scm.append((s, self.marginalize_except(list(s))))

        return inter_scm
    
    def to_regular_scm(self):
        return self

    @abstractmethod
    def consolidate(self, *args, **kwargs):
        raise NotImplementedError()

class AbstractCCV(ABC):
    @abstractmethod
    def get_inter_variables(self):
        raise NotImplementedError()

    @abstractmethod
    def get_obs_variables(self):
        raise NotImplementedError()

    @abstractmethod
    def evaluate_var(self, var, values):
        raise NotImplementedError()

    @abstractmethod
    def evaluate_vars(self, vars, values) -> dict:
        raise NotImplementedError()

class SCM(AbstractSCM):
    def __init__(self, ex_vars: dict[str, operators.StructVar]|list[operators.StructVar], en_vars: dict[str, operators.StructVar]|list[operators.StructVar]):
        if isinstance(ex_vars, dict):
            self.ex_vars = ex_vars
        elif isinstance(ex_vars, list):
            self.ex_vars = {v.id:v for v in ex_vars}
        else:
            raise ValueError("ex_vars should be either dict or list of StructVar")

        if isinstance(en_vars, dict):
            self.en_vars = en_vars
        elif isinstance(en_vars, list):
            self.en_vars = {v.id:v for v in en_vars}
        else:
            raise ValueError("en_vars should be either dict or list of StructVar")

    def evaluate_var(self, var: str, values: dict):
        running_values = copy(values)
        for exv in self.ex_vars:
            if exv not in values:
                raise ValueError("All exogenous variables must be defined")
        self.en_vars[var].evaluate(running_values)
        return running_values[var]

    def evaluate_vars(self, vars: list[str], values: dict):
        running_values = copy(values)
        for exv in self.ex_vars:
            if exv not in values:
                raise ValueError("All exogenous variables must be defined")
        for v in vars:
            self.en_vars[v].evaluate(running_values)
        return {var:running_values[var] for var in vars}

    def evaluate_rv(self, var: str, values: dict[str, rv.RandomVariable|float]):
        struct_eq = self.en_vars[var].struct_eq
        while len(struct_eq.get_variables().difference(values.keys())) > 0:
            struct_eq = struct_eq.replace(list(set(self.en_vars.keys()).difference(values.keys())))
        struct_eq = struct_eq.simplify()
        if isinstance(struct_eq, operators.Const):
            return struct_eq.value
        if isinstance(struct_eq, operators.StructVar):
            return values[struct_eq.id]
        if struct_eq.result_type == operators.ResultType.logical:
            solution = operators.solve(struct_eq)
            if isinstance(solution, operators.KDNFSolution):
                return operators.solve(struct_eq).evaluate_rv(values)
            return solution.evaluate()
        return struct_eq.evaluate(values)

    def get_parents(self, var) -> set[str]:
        return self.en_vars[var].struct_eq.get_variables()

    def get_set_parents(self, vars) -> set[str]:
        pars = set()
        for v in vars:
            pars = pars.union(self.get_parents(v))
        return pars

    def get_predecessors(self, var) -> set[str]:
        if var in self.ex_vars.keys():
            return set()
        preds = set()
        pars = self.get_parents(var)
        for p in pars:
            if p not in preds:
                preds = preds.union(self.get_predecessors(p))
        preds = preds.union(pars)
        return preds

    def get_set_predecessors(self, vars) -> set[str]:
        pred = set()
        for var in vars:
            if var not in pred:
                pred = pred.union(self.get_predecessors(var))
        return pred

    def partition(self, node_sets):
        sub_scm = []
        for part in node_sets:
            set_parents = self.get_set_parents(part)
            part_parents = [p for p in set_parents if p not in part]
            if len(set(part).intersection(self.get_set_predecessors(part_parents))) > 0:
                raise RuntimeError("Sub-scm must be well-ordered")
            part_ex_vars = {v_id:self.ex_vars[v_id] if v_id in self.ex_vars else self.en_vars[v_id] for v_id in part_parents}
            part_en_vars = {v_id:self.en_vars[v_id] for v_id in part}
            sub_scm.append(SCM(part_ex_vars, part_en_vars))
        return PartitionedSCM(sub_scm)

    def marginalize(self, vars=None):
        if vars is not None and len(vars) == 0:
            return self
        new_en_vars = {}
        for var in self.get_var_ordering():
            new_en_vars[var] = operators.StructVar(var, self.en_vars[var].struct_eq.update(new_en_vars).replace(vars).simplify())
        return SCM(copy(self.ex_vars), new_en_vars)

    def marginalize_except(self, inter_vars):
        return self.marginalize(list(self.get_en_variables().difference(inter_vars)))

    def get_var_ordering(self):
        order_map = {v:0 for v in self.ex_vars}
        to_process = list(self.ex_vars.keys())
        cg = self.get_causal_graph()
        while len(to_process) > 0:
            children = cg.get_children(to_process[0])
            to_process += list(children)
            for c in children:
                if order_map.get(c, 0) <= order_map[to_process[0]]:
                    order_map[c] = order_map[to_process[0]]+1
            to_process.remove(to_process[0])
        return sorted(list(self.en_vars.keys()), key=lambda v:order_map[v])

    def get_variables(self) -> set[str]:
        return set(self.ex_vars.keys()).union(self.en_vars.keys())

    def get_ex_variables(self) -> set[str]:
        return set(self.ex_vars.keys())

    def get_en_variables(self) -> set[str]:
        return set(self.en_vars.keys())

    def get_causal_graph(self):
        edges = []
        for var_id, var in self.en_vars.items():
            parents = var.struct_eq.get_variables()
            for p in parents:
                edges.append((p, var_id))
        return CausalGraph(list(self.ex_vars.keys()), list(self.en_vars.keys()), edges)

    def consolidate(self, observations, interventions = None):
        if observations is None:
            observations = list(self.en_vars.keys())
        if isinstance(observations, str):
            observations = [observations]
        defining_sets = []
        blocked_sets = []
        defining_eq_map = []
        ccv_vars = {}
        part_marg = sorted(self.simulate_interventions(interventions), key=lambda m: len(m[0]))
        for var in observations:
            ccv_vars[var] = part_marg[0][1].en_vars[var]
        for intervention, model in part_marg[1:]:
            is_new = True
            for i, def_set in enumerate(defining_sets):
                if def_set.issubset(set(intervention)) and set(intervention).difference(def_set).issubset(blocked_sets[i]):
                    is_new = False
                    break
            if is_new:
                obs_parents = set()
                struct_eq_map = {}
                for obs in observations:
                    obs_parents = obs_parents.union(model.get_parents(obs))
                    struct_eq_map[obs] = model.en_vars[obs]
                defining_sets.append(obs_parents.intersection(intervention))
                blocked_sets.append(self.get_causal_graph().compute_blocked(list(intervention), observations))
                defining_eq_map.append(struct_eq_map)
        return CCV(ccv_vars, copy(interventions), defining_sets, blocked_sets, defining_eq_map)

    def add_interventional_vars(self, intervened_vars):
        return GuardedIVarSCM(copy(self.ex_vars), copy(self.en_vars), allowed_interventions=intervened_vars)

    def auto_partition(self, border_vars, part_algo = part_causal_graph_foreward):
        return self.partition(part_algo(self.get_causal_graph(), border_vars))

class PartitionedSCM(AbstractSCM):
    def __init__(self, sub_scm: list[AbstractSCM]):
        self.sub_scm = sub_scm

    def evaluate_var(self, var, values):
        running_values = copy(values)
        scm_idx = self.locate(var)
        running_values.update(self.evaluate_vars(list(self.sub_scm[scm_idx].get_ex_variables()), values))
        return self.sub_scm[scm_idx].evaluate_var(var, running_values)

    def evaluate_vars(self, vars, values):
        remaining = copy(vars)
        for i, scm1 in enumerate(self.sub_scm[:-1]):
            for scm2 in self.sub_scm[i+1:]:
                values.update(scm1.evaluate_vars([ex_var for ex_var in scm2.get_ex_variables() if ex_var in scm1.get_en_variables()], values))
        for scm in self.sub_scm:
            scm_vars = [var for var in remaining if var in scm.get_en_variables()]
            values.update(scm.evaluate_vars(scm_vars, values))
            for var in scm_vars:
                remaining.remove(var)
        return {var:values[var] for var in vars}

    def evaluate_rv(self, var: str, values: dict[str, rv.RandomVariable|float]):
        return self.to_regular_scm().evaluate_rv(var, values)

    def locate(self, var):
        for i, scm in enumerate(self.sub_scm):
            if var in scm.get_en_variables():
                return i
        raise ValueError(f"{var} is not an endogenous variable in this scm")

    def marginalize(self, vars=None):
        new_sub_scm = []
        for scm in self.sub_scm:
            if vars is None:
                scm_vars = None
            else:
                scm_vars = scm.get_en_variables().intersection(vars)
            new_sub_scm.append(scm.marginalize(scm_vars))
        return PartitionedSCM(new_sub_scm)

    def marginalize_except(self, inter_vars):
        new_sub_scm = []
        for sub_scm in self.sub_scm:
            new_sub_scm.append(sub_scm.marginalize_except([var for var in inter_vars if var in sub_scm.get_en_variables()]))
        return PartitionedSCM(new_sub_scm)

    def get_causal_graph(self) -> CausalGraph:
        graph = self.sub_scm[0].get_causal_graph()
        for scm in self.sub_scm[1:]:
            to_add = scm.get_causal_graph()
            graph.ex_vars += set(to_add.ex_vars).difference(graph.en_vars)
            graph.en_vars += to_add.en_vars
            graph.edges += to_add.edges
        return graph

    def get_variables(self):
        vars = set()
        for scm in self.sub_scm:
            vars = vars.union(scm.get_variables())
        return vars

    def get_en_variables(self) -> set[str]:
        vars = set()
        for scm in self.sub_scm:
            vars = vars.union(scm.get_en_variables())
        return vars

    def get_ex_variables(self) -> set[str]:
        vars = set()
        en_vars = self.get_en_variables()
        for scm in self.sub_scm:
            vars = vars.union(scm.get_ex_variables().difference(en_vars))
        return vars

    def get_parents(self, var):
        return self.sub_scm[self.locate(var)].get_parents(var)

    def to_regular_scm(self):
        ex_var_ids = self.get_ex_variables()
        ex_vars = {}
        en_vars = {}
        for scm in self.sub_scm:
            if not isinstance(scm, (SCM)):
                scm = scm.to_regular_scm()
            ex_vars.update({ex_id:var for ex_id, var in scm.ex_vars.items() if ex_id in ex_var_ids})
            en_vars.update({en_id:var for en_id, var in scm.en_vars.items()})

        return SCM(ex_vars, en_vars)
    
    def consolidate(self, scm_idx, observations, interventions = None):
        if interventions is None:
            interventions = list(self.sub_scm[scm_idx].get_en_variables())
        sub_en_vars = self.sub_scm[scm_idx].get_en_variables()
        for i, scm in enumerate(self.sub_scm):
            if i != scm_idx:
                for ex_var in scm.get_ex_variables().intersection(sub_en_vars):
                    if ex_var not in observations:
                        observations.append(ex_var)
        new_sub_scm = []
        for i, scm in enumerate(self.sub_scm):
            if i != scm_idx:
                new_sub_scm.append(copy(scm))
            else:
                new_sub_scm.append(scm.consolidate(observations, interventions))
        return PartitionedSCM(new_sub_scm)

class GuardedIVarSCM(AbstractSCM):
    def __init__(self, ex_vars, en_vars, i_vars=None, g_vars = None, allowed_interventions=None):
        self.ex_vars = ex_vars
        if allowed_interventions is not None:
            self.i_vars = {var_id:operators.StructVar(f"i_{var_id}", None) for var_id in en_vars.keys() if allowed_interventions is not None and var_id in allowed_interventions}
            self.g_vars = {var_id:operators.StructVar(f"g_i_{var_id}", operators.Const(False)) for var_id in en_vars.keys() if allowed_interventions is not None and var_id in allowed_interventions}
            self.en_vars = {var_id:operators.StructVar(var_id, operators.CondSwitch([self.g_vars[var_id]], [self.i_vars[var_id], var.struct_eq]).simplify()) for var_id, var in en_vars.items()}
        elif isinstance(i_vars, dict) and g_vars is None:
            self.i_vars = i_vars
            self.g_vars = {f"g_{var_id}":operators.StructVar(f"g_{var_id}", operators.Const(False)) for var_id in i_vars.keys()}
            self.en_vars = en_vars
        elif isinstance(i_vars, dict) and isinstance(g_vars, dict):
            self.i_vars = i_vars
            self.g_vars = g_vars
            self.en_vars = en_vars
        else:
            raise RuntimeError("Either i_vars or allowed_interventions must be defined.")

    def evaluate_var(self, var, values: dict):
        running_values = copy(values)
        for var_id in self.i_vars:
            if var_id in running_values:
                running_values.update({f"g_{var_id}":True, f"g_i_{var_id}":running_values[var_id]})
                del running_values[var_id]
        return self.en_vars[var].evaluate(running_values)

    def evaluate_vars(self, vars, values):
        running_values = copy(values)
        for var_id in self.i_vars:
            if var_id in running_values:
                running_values.update({f"g_{var_id}":True, f"g_i_{var_id}":running_values[var_id]})
                del running_values[var_id]
        for var in vars:
            self.en_vars[var].evaluate(running_values)
        return {var_id:value for var_id, value in running_values.items()}

    def get_causal_graph(self) -> CausalGraph:
        edges = []
        for var_id, var in self.en_vars.items():
            parents = var.struct_eq.get_variables()
            for p in parents:
                if p in self.ex_vars or p in self.en_vars:
                    edges.append((p, var_id))
        return CausalGraph(list(self.ex_vars.keys()), list(self.en_vars.keys()), edges)

    def get_parents(self, var):
        if var in self.ex_vars:
            return set()
        return set(self.en_vars[var].struct_eq.get_variables()).intersection(self.get_variables())

    def get_set_parents(self, vars):
        pars = set()
        for v in vars:
            pars = pars.union(self.get_parents(v))
        return pars

    def get_predecessors(self, var):
        preds = set()
        pars = self.get_parents(var)
        for p in pars:
            if p not in preds:
                preds = preds.union(self.get_predecessors(p))
        preds = preds.union(pars)
        return preds

    def get_set_predecessors(self, vars):
        pred = set()
        for var in vars:
            if var not in pred:
                pred = pred.union(self.get_predecessors(var))
        return pred

    def get_var_ordering(self):
        order_map = {v:0 for v in self.ex_vars}
        to_process = list(self.ex_vars.keys())
        cg = self.get_causal_graph()
        while len(to_process) > 0:
            children = cg.get_children(to_process[0])
            to_process += list(children)
            for c in children:
                if order_map.get(c, 0) <= order_map[to_process[0]]:
                    order_map[c] = order_map[to_process[0]]+1
            to_process.remove(to_process[0])
        return sorted(list(self.en_vars.keys()), key=lambda v:order_map[v])

    def partition(self, node_sets):
        sub_scm = []
        for part in node_sets:
            set_parents = self.get_set_parents(part)
            part_parents = [p for p in set_parents if p not in part]
            if len(set(part).intersection(self.get_set_predecessors(part_parents))) > 0:
                raise RuntimeError("Sub-scm must be well-ordered")
            part_ex_vars = {v_id:self.ex_vars[v_id] if v_id in self.ex_vars else self.en_vars[v_id] for v_id in part_parents}
            part_en_vars = {v_id:self.en_vars[v_id] for v_id in part}
            i_vars = set()
            g_vars = set()
            for var_id, var in part_en_vars.items():
                var_parents = var.struct_eq.get_variables()
                i_vars = i_vars.union(var_parents.intersection(self.i_vars.keys()))
                g_vars = g_vars.union(var_parents.intersection(self.g_vars.keys()))
            part_i_vars = {v_id:self.i_vars[v_id] for v_id in part if v_id in self.i_vars}
            part_g_vars = {v_id:self.g_vars[v_id] for v_id in part if v_id in self.g_vars}
            sub_scm.append(GuardedIVarSCM(part_ex_vars, part_en_vars, part_i_vars, part_g_vars))
        return PartitionedSCM(sub_scm)

    def marginalize(self, vars=None):
        new_en_vars = {}
        for var in self.get_var_ordering():
            new_en_vars[var] = operators.StructVar(var, self.en_vars[var].struct_eq.update(new_en_vars).replace(vars).simplify())
        return GuardedIVarSCM(copy(self.ex_vars), new_en_vars, copy(self.i_vars), copy(self.g_vars))

    def marginalize_except(self, inter_vars):
        return self.marginalize(list(set(self.en_vars.keys()).difference(inter_vars)))

    def get_variables(self) -> set[str]:
        return set(self.en_vars.keys()).union(self.ex_vars.keys())

    def get_en_variables(self) -> set[str]:
        return set(self.en_vars.keys())

    def get_ex_variables(self) -> set[str]:
        return set(self.ex_vars.keys())
    
    def consolidate(self, observations, _):
        marg_model = self.marginalize()
        ex_vars = copy(marg_model.ex_vars)
        en_vars = {idx:var for idx, var in self.en_vars.items() if idx in observations}
        i_vars = copy(marg_model.i_vars)
        self.g_vars = copy(marg_model.g_vars)

        return GuardedIVarSCM(ex_vars, en_vars, i_vars, self.g_vars)
    
    def to_regular_scm(self):
        new_en_vars = {}
        for key, var in self.en_vars.items():
            if isinstance(var.struct_eq, operators.CondSwitch) and len(var.struct_eq.get_variables().intersection(self.i_vars.keys())) > 0:
                new_en_vars[key] = operators.StructVar(key, var.struct_eq.cases[-1])
            else:
                new_en_vars[key] = copy(var)

        return SCM(copy(self.ex_vars), new_en_vars)
    
    def auto_partition(self, border_vars, part_algo = part_causal_graph_foreward):
        return self.partition(part_algo(self.get_causal_graph(), border_vars))

class CCV(AbstractCCV):
    def __init__(self, observations, intervened_vars, defining_vars, defining_blocked_map, defining_eq_map):
        self.observations = observations
        self.intervened_vars = intervened_vars
        self.defining_blocked_map = defining_blocked_map
        self.defining_eq_map = defining_eq_map
        self.defining_vars = defining_vars

    def evaluate_var(self, var, values):
        inter_set = set(values.keys()).intersection(self.intervened_vars)
        if len(inter_set) == 0:
            return self.observations[var].evaluate(values)
        for i, def_set in reversed(list(enumerate(self.defining_vars))):
            if def_set.issubset(inter_set) and inter_set.difference(def_set).issubset(self.defining_blocked_map[i]):
                return self.defining_eq_map[i][var].evaluate(values)
        raise ValueError(f"Something went wrong while evaluating {var}")

    def evaluate_vars(self, vars, values):
        inter_set = set(values.keys()).intersection(self.intervened_vars)
        solutions = {}
        if len(inter_set) == 0:
            for idx in vars:
                solutions[idx] = self.observations[idx].evaluate(values)
            return solutions
        for i, def_set in enumerate(self.defining_vars):
            if def_set.issubset(inter_set) and inter_set.difference(def_set).issubset(self.defining_blocked_map[i]):
                for idx in vars:
                    solutions[idx] = self.defining_eq_map[i][idx].evaluate(values)
                return solutions
        raise ValueError(f"Something went wrong while evaluating {vars}")
    
    def evaluate_rv(self, var, values):
        return self.evaluate_var(var, values)

    def get_obs_variables(self):
        return set(self.observations.keys())

    def get_inter_variables(self):
        return set(self.intervened_vars)

def scm_from_matrices(ex_vars: list[str], en_vars: list[str], ex_matrix: np.ndarray, en_matrix: np.ndarray, const_terms: np.ndarray):
    ex_var_dict = {key:operators.StructVar(key, None) for key in ex_vars}
    en_var_dict = {}
    for idx, var in enumerate(en_vars):
        params = []
        variables = {}
        positions = {}
        position = 0
        for fac_idx, fac_var in enumerate(ex_vars):
            if ex_matrix[idx, fac_idx] != 0:
                params.append(ex_matrix[idx, fac_idx])
                variables[fac_var] = ex_var_dict[fac_var]
                positions[fac_var] = position
                position += 1
        for fac_idx, fac_var in enumerate(en_vars):
            if en_matrix[idx, fac_idx] != 0:
                params.append(en_matrix[idx, fac_idx])
                variables[fac_var] = en_var_dict[fac_var]
                positions[fac_var] = position
                position += 1
        params.append(const_terms[idx])
        en_var_dict[var] = operators.StructVar(var, operators.LinearFunction(variables, params, positions))
    return SCM(ex_var_dict, en_var_dict)

def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
