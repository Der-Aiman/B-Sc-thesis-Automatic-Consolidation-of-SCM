"""
This module provides an example for definition of an SCM
"""

import causal_models
import operators
import random_variable
import numpy as np
from copy import copy
import random_variable as rv
from matplotlib import pyplot as plt

layer_n = [8, 5, 4]
layer_base_rates = [2, 3, 4]
layer_randomness = [rv.ExponentialDistRV(1/2), rv.ExponentialDistRV(1/3), rv.ExponentialDistRV(1/4)]

random_values = {f"layer_0_dist_{j}":copy(layer_randomness[0]) for j in range(layer_n[0])}
ex_vars = {f"layer_0_dist_{j}":operators.StructVar(f"layer_0_dist_{j}", None) for j in range(layer_n[0])}
en_vars = {f"layer_0_var_0":operators.StructVar(f"layer_0_var_0", operators.LinearFunction(copy(ex_vars), [1.0 for _ in range(layer_n[0])] + [layer_n[0]*layer_base_rates[0]], {key:pos for pos, key in enumerate(ex_vars.keys())}))}
layer_ex_vars = {}
for i, n in list(enumerate(layer_n))[1:]:
    layer_ex_vars = {f"layer_{i}_dist_{j}":operators.StructVar(f"layer_{i}_dist_{j}", None) for j in range(n)}
    ex_vars.update(layer_ex_vars)
    en_vars.update({f"layer_{i}_var_0":operators.StructVar(f"layer_{i}_var_0", operators.Min([en_vars[f"layer_{i-1}_var_0"], operators.LinearFunction(copy(layer_ex_vars), [1.0 for _ in range(n)] + [n*layer_base_rates[i]], {key:pos for pos, key in enumerate(layer_ex_vars.keys())})]))})
    random_values.update({f"layer_{i}_dist_{j}":copy(layer_randomness[i]) for j in range(n)})

scm = causal_models.SCM(ex_vars, en_vars)

dist = scm.evaluate_rv("layer_2_var_0", random_values)

#vals = []

#lb = 0
#ub = 200

#for i in range(1001):
#    x = lb + i * ((ub - lb)/1000)
#    vals.append((x, dist.get_pdf()(x)))

#plot, ax = plt.subplots(1, 1)
#ax.plot([pair[0] for pair in vals], [pair[1] for pair in vals])

#plt.savefig("/path/to/file.pdf")