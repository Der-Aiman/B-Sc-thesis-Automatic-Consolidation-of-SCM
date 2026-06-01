"""
This module provides functionality for function approximation.
"""

import numpy as np
import sympy

# weights used for approximation of integrals
NC_WEIGHTS = [
    [1/2, 1/2],
    [1/6, 2/3, 1/6],
    [1/8, 3/8, 3/8, 1/8],
    [7/90, 16/45, 2/15, 16/45, 7/90],
    [19/288, 25/96, 25/144, 25/144, 25/96, 19/288],
    [41/840, 9/35, 9/280, 34/105, 9/280, 9/35, 41/840],
    [751/17280, 3577/17280, 49/640, 2989/17280, 2989/17280, 49/640, 3577/17280, 751/17280]
    ]

class Spline4:
    """
    This class represents splines, using segments made up of fourth-degree polynonials.
    Usually represents the antiderivative of a cubic spline.
    """
    def __init__(self, points, params):
        self.points = points
        self.params = params

    def __call__(self, x):
        i = 0
        while i < len(self.points) and x > self.points[i][0]:
            i += 1
        if i == 0:
            return self.points[0][1]
        if i == len(self.points):
            return self.points[-1][1]
        if x == self.points[i][0]:
            return self.points[i][1]
        return self.params[i-1][0] * x**4 + self.params[i-1][1] * x**3 + self.params[i-1][2] * x**2 + self.params[i-1][3] * x + self.params[i-1][4]

    def normalize(self):
        """
        Rescales the function and translates it such that it is 0 at the lower bound
        and 1 at the upper bound.
        """
        offset = self.points[0][1]
        mult = 1/(self.points[-1][1] - offset)
        new_params = []
        new_points = [(x, (y - offset)*mult) for x, y in self.points]
        for comp in self.params:
            new_comp = [p*mult for p in comp[:4]]
            new_comp.append((comp[4] - offset)*mult)
            new_params.append(new_comp)
        return Spline4(new_points, new_params)

    def lift(self, offset):
        """
        Increases the function value by offset at every point
        """
        for comp in self.params:
            comp[4] += offset

    def scale(self, factor):
        """
        Multiplies the function value by factor at every point
        """
        for comp in self.params:
            for i, fac in enumerate(comp):
                comp[i] = fac*factor

    def to_sympy_expr(self):
        """
        Returns a sympy expression that describes the represented function
        """
        components = [(self.points[0][1], sympy.LessThan(sympy.symbols("x"), self.points[0][0]))]
        for i, comp in enumerate(self.params):
            components.append((sympy.sympify(f"{comp[0]} * x**4 + {comp[1]} * x**3 + {comp[2]} * x**2 + {comp[3]} * x + {comp[4]}"), sympy.And(sympy.StrictGreaterThan(sympy.symbols("x"), self.points[i][0]), sympy.LessThan(sympy.symbols("x"), self.points[i+1][0]))))
        components.append((self.points[-1][1], sympy.StrictGreaterThan(sympy.symbols("x"), self.points[-1][0])))
        return sympy.Piecewise(*components)

class Spline3:
    """
    This class represents cubic splines used for function approximation
    """
    def __init__(self, points, moments, C, D):
        self.points = points
        self.moments = moments
        self.C = C
        self.D = D

    def __call__(self, x):
        i = 0
        while i < len(self.points) and x > self.points[i][0]:
            i += 1
        if i == 0:
            return self.points[0][1]
        if i == len(self.points):
            return self.points[-1][1]
        if i == self.points[i][0]:
            return self.points[i][1]
        return (((self.points[i][0] - x)**3/(self.points[i][0] - self.points[i-1][0]))*self.moments[i-1]+((x - self.points[i-1][0])**3/(self.points[i][0] - self.points[i-1][0]))*self.moments[i]) + self.C[i-1]*(x - self.points[i-1][0]) + self.D[i-1]

    def to_sympy_expr(self):
        """
        Returns a sympy expression that describes the represented function
        """
        components = [(self.points[0][1], sympy.LessThan(sympy.symbols("x"), self.points[0][0]))]
        for i in range(len(self.C)):
            components.append((sympy.sympify(f"((({self.points[i+1][0]} - x)**3/({self.points[i+1][0]} - {self.points[i][0]}))*{self.moments[i]} + ((x - {self.points[i][0]})**3/({self.points[i+1][0]} - {self.points[i][0]}))*{self.moments[i+1]}) + {self.C[i]}*(x - {self.points[i][0]}) + {self.D[i]}"), sympy.And(sympy.StrictGreaterThan(sympy.symbols("x"), self.points[i][0]), sympy.LessThan(sympy.symbols("x"), self.points[i+1][0]))))
        components.append((self.points[-1][1], sympy.StrictGreaterThan(sympy.symbols("x"), self.points[-1][0])))
        return sympy.Piecewise(*components)

    def get_antiderivative(self, offset=0):
        """
        Returns the antiderivative of the represented function as a fourth-degree spline.
        Offset represents the antiderivative's value at the lower bound
        """
        comp_params = [(self.moments[1] - self.moments[0])/(24*(self.points[1][0]-self.points[0][0])), (3*self.moments[0]*self.points[1][0]-3*self.moments[1]*self.points[0][0])/(18*(self.points[1][0] - self.points[0][0])), ((3*self.moments[1]*self.points[0][0]**2 - 3*self.moments[0]*self.points[1][0]**2)/(6*(self.points[1][0] - self.points[0][0])) + self.C[0])/2, (self.moments[0]*self.points[1][0]**3 - self.moments[1]*self.points[0][0]**3)/(6*(self.points[1][0] - self.points[0][0])) - self.C[0]*self.points[0][0] + self.D[0]]
        comp_params.append(offset - (comp_params[0]*self.points[0][0]**4 + comp_params[1]*self.points[0][0]**3 + comp_params[2]*self.points[0][0]**2 + comp_params[3]*self.points[0][0]))
        params = [tuple(comp_params)]
        ad_points = [(self.points[0][0], offset), (self.points[1][0], comp_params[0]*self.points[1][0]**4 + comp_params[1]*self.points[1][0]**3 + comp_params[2]*self.points[1][0]**2 + comp_params[3]*self.points[1][0] + comp_params[4])]
        for i in range(len(self.C)-1):
            ii = i + 1
            comp_params = [(self.moments[ii+1] - self.moments[ii])/(24*(self.points[ii+1][0]-self.points[ii][0])), (3*self.moments[ii]*self.points[ii+1][0]-3*self.moments[ii+1]*self.points[ii][0])/(18*(self.points[ii+1][0] - self.points[ii][0])), ((3*self.moments[ii+1]*self.points[ii][0]**2 - 3*self.moments[ii]*self.points[ii+1][0]**2)/(6*(self.points[ii+1][0] - self.points[ii][0])) + self.C[ii])/2, (self.moments[ii]*self.points[ii+1][0]**3 - self.moments[ii+1]*self.points[ii][0]**3)/(6*(self.points[ii+1][0] - self.points[ii][0])) - self.C[ii]*self.points[ii][0] + self.D[ii]]
            comp_params.append(ad_points[-1][1] - (comp_params[0]*self.points[ii][0]**4 + comp_params[1]*self.points[ii][0]**3 + comp_params[2]*self.points[ii][0]**2 + comp_params[3]*self.points[ii][0]))
            params.append(tuple(comp_params))
            ad_points.append((self.points[ii+1][0], comp_params[0]*self.points[ii+1][0]**4 + comp_params[1]*self.points[ii+1][0]**3 + comp_params[2]*self.points[ii+1][0]**2 + comp_params[3]*self.points[ii+1][0] + comp_params[4]))
        return Spline4(ad_points, params)

    def shift(self, value):
        """
        Shifts the represented function by value such that f'(x) = f(x - value)
        where f' is the shifted function and f is the original function
        """
        new_points = [(x+value, y) for x, y in self.points]
        return Spline3(new_points, self.moments, self.C, self.D)

    def stretch_antid(self, factor):
        """
        Modifies the function such that its antiderivative will be stretched
        in the x-direction.
        Usually used when the spline approximates a probability distribution
        """
        new_points = ((factor*x, y) for x, y in self.points)
        new_moments = [m/(factor**3) for m in self.moments]
        new_C = [c/(factor**2) for c in self.C]
        new_D = [d/factor for d in self.D]
        return Spline3(new_points, new_moments, new_C, new_D)

    def get_derivative(self):
        """
        Returns the derivative of the represented function as a quadratic spline
        """
        new_points = [(point[0], -((self.points[i+1][0] - point[0])/2)*self.moments[i] + self.C[i]) for i, point in enumerate(self.points[:-1])]
        new_points.append((self.points[-1][0], ((self.points[-1][0] - self.points[-2][0])/2)*self.moments[-1] + self.C[-1]))
        return Spline2(new_points, self.moments, self.C)

    def get_local_params(self, x):
        # for segment lower bound < x <= segment upper bound
        # M0, M1, c, d, segment lower bound, segment upper bound
        if not (self.points[0][0] <= x <= self.points[-1][0]):
            raise ValueError(f"{x} is out of bounds for Spline3({self.points[0][0]}, {self.points[-1][0]})")
        i = 0
        while i < len(self.C):
            if x > self.points[i+1][0]:
                i += 1
            else:
                break
        return (self.moments[i], self.moments[i+1], self.C[i], self.D[i], self.points[i][0], self.points[i+1][0])

class Spline2:
    """
    This class represents quadratic splines.
    Usually represents the antiderivative of a cubic spline.
    """
    def __init__(self, points, moments, C):
        self.points = points
        self.moments = moments
        self.C = C

    def __call__(self, x):
        i = 0
        while x > self.points[i][0] and i < len(self.points):
            i += 1
        if i == 0:
            return self.points[0][1]
        if i == len(self.points):
            return self.points[-1][1]
        if i == self.points[i][0]:
            return self.points[i][1]
        return (-((self.points[i][0] - x)**2/(self.points[i][0] - self.points[i-1][0]))*self.moments[i-1] + ((x - self.points[i-1][0])**2/(self.points[i] - self.points[i-1]))*self.moments[i])/2 + self.C[i-1]

    def to_sympy_expr(self):
        """
        Returns a sympy expression that describes the represented function
        """
        components = [(self.points[0][1], sympy.LessThan(sympy.symbols("x"), self.points[0][0]))]
        for i in range(len(self.C)):
            components.append(sympy.sympify(f"-(({self.points[i+1][0]} - x)**2/({self.points[i+1][0] - self.points[i][0]}))*{self.moments[i]} + ((x - {self.points[i][0]})**2/({self.points[i+1] - self.points[i]}))*{self.moments[i+1]})/2 + {self.C[i]}"))

def compute_spline3(points):
    """
    Computes a cubic spline approximating a function that includes
    the specified points
    """
    H = [points[i+1][0] - points[i][0] for i in range(len(points)-1)]
    parameters = [[1, 0] + [0 for _ in range(len(points) - 2)]]
    values = [0]
    for i in range(len(points) - 2):
        line = [0 for _ in range(len(points))]
        line[i] = H[i]/6
        line[i+1] = (H[i]+H[i+1])/3
        line[i+2] = H[i+1]/6
        parameters.append(line)
        values.append(((points[i+2][1]-points[i+1][1])/H[i+1])-((points[i+1][1]-points[i][1])/H[i]))
    parameters.append([0 for _ in range(len(points) - 2)] + [0, 1])
    values.append(0)

    moments = np.linalg.solve(np.array(parameters), np.array(values))

    C = []
    D = []
    for i in range(len(points)-1):
        D.append(points[i][1] - (H[i]**2/6)*moments[i])
        C.append((points[i+1][1]-D[i])/H[i] - (H[i]/6)*moments[i+1])
    return Spline3(points, moments, C, D)

def newton_cotesn(function, bounds, degree):
    """
    Approximates the integral of the specified function over the
    specified bounds using a newton-cotes formula with the specified degree.
    """
    if not 0 < degree <= 7:
        raise ValueError("degree must be between 1 and 7 (including 1 and 7)")
    h = bounds[1] - bounds[0]
    val = 0
    for i in range(degree+1):
        val += h*NC_WEIGHTS[degree-1][i]*function(bounds[0]+i*h/degree)
    return val

def fixed_intervals_summed_newton_cotes(function, bounds, intervals, degree):
    """
    Approximates the integral of the specified function over the specified bounds,
    splitting up computation over intervals segments using a summed newton-cotes
    formula of the specified degree
    """
    h = (bounds[1] - bounds[0])/intervals
    lower_bound = bounds[0]
    acc_val = 0
    for i in range(intervals):
        acc_val += newton_cotesn(function, (lower_bound, lower_bound + h), degree)
        lower_bound += h
    return acc_val
