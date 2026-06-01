"""
This module provides functionality for random variable arithmetic
"""

from abc import ABC
from typing import Callable
from copy import copy
from math import log, e
import random
import sympy
import approx_utils

# global variables
approx_samples = 1000
sum_nc_samples = 500
approx_threshold = 0.001

class GenericFunction:
    """
    This class represents a function, represented by a sympy expression
    """
    def __init__(self, expr: sympy.Expr, variable: str):
        self.expr = expr
        self.variable = variable

    def __call__(self, x):
        return float(self.expr.subs(sympy.symbols(self.variable), x).evalf())

    def subs(self, substitutions):
        return GenericFunction(self.expr.subs(substitutions), str(dict(substitutions)[sympy.symbols(self.variable)]))

    def evalf(self):
        return float(self.expr.evalf())

    def diff(self):
        return GenericFunction(self.expr.diff(), self.variable)

class PiecewiseSplinePDF:
    def __init__(self, spline: approx_utils.Spline3, lb: float =-float("inf"), ub: float =float("inf")):
        self.lb = lb
        self.spline = spline
        self.ub = ub

    def __call__(self, x):
        if x < self.lb or x > self.ub:
            return 0
        if x == self.lb or x == self.ub:
            return float("inf")
        return self.spline(x)

    def get_cdf(self):
        def cdf(x):
            if x < self.lb:
                return 0
            if x > self.ub:
                return 1
            return self.spline.get_antiderivative().normalize()(x)
        return cdf

class RandomVariable(ABC):
    """
    This class provides an interface for classes implementing random variables
    """
    pdf: Callable

    def sample(self):
        raise NotImplementedError()

    def get_domain(self):
        pass

    def get_pdf(self):
        return self.pdf

class ContinuousRV(RandomVariable):
    """
    This class represents generic continuous random variables.
    It is advised to use other classes to define random variables
    """
    def __init__(self, pdf: GenericFunction|None =None, cdf: GenericFunction|None =None):
        if (pdf is None) & (cdf is None):
            raise ValueError("pdf or cdf must be given.")
        if (pdf is not None) & (cdf is not None):
            self.pdf = pdf
            self.cdf = cdf
        if pdf is not None:
            self.pdf = pdf
            self.cdf = GenericFunction(sympy.Integral(pdf.expr, (sympy.symbols(self.pdf.variable), -sympy.oo, sympy.symbols(self.pdf.variable+ "'"))), self.pdf.variable+ "'")
        if cdf is not None:
            self.cdf = cdf
            self.pdf = GenericFunction(sympy.diff(cdf, sympy.symbols(self.cdf.variable)), self.cdf.variable)

    def get_domain(self):
        return float

    def get_cdf(self):
        return self.cdf

    def __add__(self, other):
        if isinstance(other, float):
            return ContinuousRV(cdf = self.cdf.subs((sympy.symbols(self.cdf.variable), sympy.symbols(self.cdf.variable) - other)))
        if isinstance(other, ContinuousRV):
            raise RuntimeError("Cannot add general random variables")

    def to_bounded_rv(self, threshold = approx_threshold, samples = approx_samples):
        raise RuntimeError("Conversion of general random variable to approximation is impossible")

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

class ApproxContinuousRV(ContinuousRV):
    """
    This class represents continuous random variables whose pdf
    is approximated by a cubic spline
    """
    def __init__(self, pdf: approx_utils.Spline3|PiecewiseSplinePDF):
        self.pdf = pdf

    def get_pdf(self):
        return self.pdf

    def get_cdf(self):
        if isinstance(self.pdf, approx_utils.Spline3):
            return self.pdf.get_antiderivative().normalize()
        return self.pdf.get_cdf()

    def __add__(self, other):
        if isinstance(other, (float, int)):
            return ApproxContinuousRV(self.pdf.shift(other))
        if isinstance(other, ApproxContinuousRV):
            lb = self.pdf.points[0][0] + other.pdf.points[0][0]
            ub = self.pdf.points[-1][0] + other.pdf.points[-1][0]
            new_points = [(lb, 0)]
            for i in range(approx_samples):
                x = lb + (ub - lb)*(i+1)/(approx_samples+1)
                bounds = (max(self.get_pdf().points[0][0], x - other.get_pdf().points[-1][0]), min(self.get_pdf().points[-1][0], x - other.get_pdf().points[0][0]))
                new_points.append((x, approx_utils.fixed_intervals_summed_newton_cotes(lambda v: self.get_pdf()(v)*other.get_pdf()(x-v), bounds, sum_nc_samples, 7)))
            new_points.append((ub, 0))
            return ApproxContinuousRV(approx_utils.compute_spline3(new_points))
        if isinstance(other, ContinuousRV):
            return self + other.to_bounded_rv()
        raise RuntimeError(f"Something went wrong while trying to add {self} and {other}")

    def __mul__(self, other):
        if isinstance(other, (float, int)):
            if other == 0:
                return 0
            return ApproxContinuousRV(self.pdf.stretch_antid(other))
        raise RuntimeError(f"Something went wrong while trying to multiply {self} and {other}")

    @classmethod
    def min(cls, *args):
        if len(args) == 1:
            return args[0]
        approx_rv = []
        for el in args:
            if isinstance(el, ApproxContinuousRV):
                approx_rv.append(el)
            elif isinstance(el, ContinuousRV):
                approx_rv.append(el.to_bounded_rv())
            else:
                raise TypeError("Only ContinuousRV")

        lb = min(*[rv.get_pdf().points[0][0] for rv in approx_rv])
        ub = min(*[rv.get_pdf().points[-1][0] for rv in approx_rv])

        def min_pdf(x):
            acc_val = 0
            for i, i_var in enumerate(approx_rv):
                part_diff = 1
                for j, j_var in enumerate(approx_rv):
                    if (1 - j_var.get_cdf()(x) if i != j else i_var.get_pdf()(x)) < 0 or (1 - j_var.get_cdf()(x) if i != j else i_var.get_pdf()(x)) > 1:
                        print(f"{i};{j} --- {1 - j_var.get_cdf()(x) if i != j else i_var.get_pdf()(x)}")
                    term = 1 - j_var.get_cdf()(x) if i != j else i_var.get_pdf()(x)
                    if term < approx_threshold:
                        part_diff = 0
                        break
                    if term >= 1 - approx_threshold:
                        term = 1
                    part_diff *= term
                acc_val += part_diff
            return acc_val

        points = [(lb, 0)]
        for i in range(approx_samples):
            x = lb + (ub - lb)*(i+1)/(approx_samples+1)
            points.append((x, min_pdf(x)))
        points.append((ub, 0))
        min_rv = ApproxContinuousRV(approx_utils.compute_spline3(points))

        return min_rv

class ExponentialDistRV(ContinuousRV):
    """
    This class represents random variables with an exponential distribution
    """
    def __init__(self, lamb: float):
        if lamb <= 0:
            raise ValueError("Lambda must be greater than 0")
        self.lamb = lamb

        x = sympy.symbols("x")
        self.cdf = GenericFunction(sympy.Piecewise((0, sympy.StrictLessThan(x, 0)), (1-sympy.E**(-self.lamb*x), sympy.GreaterThan(x, 0))), "x")
        self.pdf = GenericFunction(sympy.Piecewise((0, sympy.StrictLessThan(x, 0)), (self.lamb*sympy.E**(-self.lamb*x), sympy.GreaterThan(x, 0))), "x")

    def get_pdf(self):
        return self.pdf

    def get_cdf(self):
        return self.cdf

    def sample(self):
        return random.expovariate(self.lamb)

    def __add__(self, other):
        if isinstance(other, ExponentialDistRV):
            lb = 0
            ub = -log(approx_threshold)/self.lamb - log(approx_threshold)/other.lamb
            if self.lamb == other.lamb:
                def new_pdf(x):
                    return (self.lamb**2)*x*(e**(-self.lamb*x))
            else:
                def new_pdf(x):
                    return ((self.lamb*other.lamb)/(other.lamb - self.lamb))*(e**(-self.lamb*x) - e**(-other.lamb*x))

            points = [(float(lb), 0)]
            for i in range(approx_samples):
                x = lb + (ub - lb)*(i+1)/(approx_samples+1)
                points.append((x, new_pdf(x)))
            points.append((ub, 0))
            return ApproxContinuousRV(approx_utils.compute_spline3(points))
        if isinstance(other, ApproxContinuousRV):
            return other + self
        if isinstance(other, (float, int)):
            if other == 0:
                return self
            return self.to_bounded_rv() + other
        raise RuntimeError(f"Something went wrong while trying to add {self} and {other}")

    def __mul__(self, other):
        if isinstance(other, (float, int)):
            if other == 0:
                return 0
            return ExponentialDistRV(self.lamb/other)
        raise ValueError("random variables can only be multiplied with scalar constants")

    def to_bounded_rv(self, threshold = approx_threshold, samples = approx_samples):
        if threshold >= 1:
            raise ValueError("Something about threshold")
        lb = 0
        ub = -log(threshold)/self.lamb
        points = [(float(lb), 0)]
        for i in range(samples):
            x = lb + (ub - lb)*(i+1)/(samples+1)
            points.append((x, self.pdf(x)))
        points.append((ub, 0))
        return ApproxContinuousRV(approx_utils.compute_spline3(points))
    
    @classmethod
    def min(cls, *args):
        new_lamb = 0
        for var in args:
            if not isinstance(var, ExponentialDistRV):
                raise TypeError(f"Only inputs of type ExponentialDistRV are permitted")
            new_lamb += var.lamb
        return ExponentialDistRV(new_lamb)

class DiscreteRV(RandomVariable):
    """
    This class represent discrete random variables
    """
    def __init__(self, pdf=None, domain=int):
        if isinstance(pdf, dict):
            is_int = True
            for k in pdf.keys():
                if not isinstance(k, int):
                    is_int = True
                    break
            if not is_int:
                self.domain = set(pdf.keys())
                pdf_dict = copy(pdf)
                self.pdf = lambda x: pdf_dict.get(x, 0)
            else:
                probs = []
                for k, p in pdf.values():
                    probs.append((p, sympy.Eq(sympy.symbols("x"), k)))
                self.pdf = GenericFunction(sympy.Piecewise(*probs), "x")
        elif isinstance(pdf, GenericFunction):
            if not isinstance(GenericFunction.expr, sympy.Piecewise):
                raise TypeError(f"pdf should contain object of class {sympy.Piecewise}")
            self.pdf = pdf
            self.domain = domain

    def get_domain(self):
        return self.domain

    def get_pdf(self):
        return self.pdf

    def __eq__(self, other):
        if not isinstance(other, RandomVariable):
            return LogicalRV(self.pdf(other))

class LogicalRV(DiscreteRV):
    """
    This class represents logical random variables
    """
    def __init__(self, true_prob):
        self.true_prob = true_prob

    def get_pdf(self):
        return lambda x: self.true_prob if x else 1-self.true_prob

    def get_domain(self):
        return {True, False}

    def sample(self):
        return random.random() <= self.true_prob

    def __and__(self, other):
        if isinstance(other, LogicalRV):
            return LogicalRV(self.true_prob*other.true_prob)
        if isinstance(other, bool) and other:
            return LogicalRV(self.true_prob)
        if isinstance(other, bool) and not other:
            return False
        raise ValueError(f"cannot do LogicalRV and {type(other)}")

    def __or__(self, other):
        if isinstance(other, LogicalRV):
            return LogicalRV(1 - (1 - self.true_prob)*(1 - other.true_prob))
        if isinstance(other, bool) and other:
            return True
        if isinstance(other, bool) and not other:
            return LogicalRV(self.true_prob)

    def __ror__(self, other):
        return self | other

    def __invert__(self):
        return LogicalRV(1 - self.true_prob)

    def __xor__(self, other):
        return (self & ~other) | (~self & other)

    def __eq__(self, other):
        if isinstance(other, LogicalRV):
            return LogicalRV(self.true_prob*other.true_prob + (1 - self.true_prob)*(1 - other.true_prob))
        if isinstance(other, bool) & other:
            return LogicalRV(self.true_prob)
        if isinstance(other, bool) & ~other:
            return LogicalRV(1 - self.true_prob)

def rv_min(*args):
    """
    Returns a random variable representing the minimumm of args
    """
    remaining_args = []
    ex_dist_vars = []
    min_const = float("inf")
    min_rv = None
    for v in args:
        if isinstance(v, ExponentialDistRV):
            ex_dist_vars.append(v)
        elif isinstance(v, (float, int)) and (v < min_const):
            min_const = v
        else:
            remaining_args.append(v)

    if len(ex_dist_vars) > 0:
        min_rv = ExponentialDistRV.min(*ex_dist_vars)
    if len(remaining_args) >= 1:
        if min_rv is not None:
            remaining_args.append(min_rv)
        min_rv = ApproxContinuousRV.min(*remaining_args)
    if min_const < float("inf") and min_rv is not None:
        cdf = min_rv.get_cdf()
        if cdf(min_const) >= 1 - approx_threshold:
            return min_rv
        if cdf(min_const) <= 0 + approx_threshold:
            return min_const
        old_pdf = None
        if isinstance(min_rv, ApproxContinuousRV):
            old_pdf = min_rv.get_pdf()
        else:
            old_pdf = min_rv.to_bounded_rv().get_pdf()
        new_pdf = PiecewiseSplinePDF(old_pdf, min_const)
        return ApproxContinuousRV(new_pdf)
    elif min_const < float("inf"):
        return min_const
    if min_rv is None:
        raise RuntimeError(f"No minimum could be computed from {args}")
    return min_rv
