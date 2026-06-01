"""
This module provides an example for definition of linear SCM.
The model is loosely based on the article at
https://volta.foundation/battery-manufacturing-basics-from-catls-cell-production-line-part-1/
Values are not accurate. The example is meant solely for demonstration purposes.
"""

import numpy as np

import operators
import causal_models

matrix_constants = {
    "base_viscosity":10000,
    "base_slurry_density":1.4,
    "base_thickness":7,
    "base_uniformity":1,
    "base_porosity":0,
    "base_sheet_density":0,
    "base_tortuosity":0,
    "base_edge_quality":1,
    "base_optics":0.5,
    "environment-viscosity":0.001,
    "mixing_time-viscosity":-0.1,
    "mixer_state-viscosity":1,
    "environment-density":0.01,
    "coater_state-thickness":1,
    "speed-thickness":-0.1,
    "temperature-thickness":-0.1,
    "viscosity-thickness":-0.1,
    "slurry_density-thickness":-0.1,
    "calendering_state-porosity":1,
    "pressure-porosity":0.1,
    "thickness-porosity":-1,
    "uniformity-porosity":1,
    "calendering_state-tortuosity":1,
    "pressure-tortuosity":0.01,
    "thickness-tortuosity":-0.01,
    "electrode_maker-edge_quality":1,
    "porosity-edge_quality":-0.5,
    "sheet_density-edge_quality":-0.1,
    "tortuosity-edge_quality":0.5,
    "edge_quality-optics":1,
    "base_mixing_cost":0,
    "mixing_time-mixing_cost":0.000001,
    "base_coating_cost":0,
    "speed-coating_cost":0.00001,
    "temp-coating_cost":0.00001,
    "base_calendering_cost":0,
    "pressure-calendering_cost":0.000001,
    "base_electrode_making_cost":0
    }

ex_vars = {
    "environment":operators.StructVar("environment", None),
    "mixing_time":operators.StructVar("mixing_time", None),
    "coater_deviation":operators.StructVar("coater_deviation", None),
    "speed":operators.StructVar("speed", None),
    "temp":operators.StructVar("temp", None),
    "calendering_deviation":operators.StructVar("calendering_deviation", None),
    "pressure":operators.StructVar("pressure", None),
    "electrode_maker_accuracy":operators.StructVar("electrode_maker_accuracy", None)
}

ex_var_list = ["environment", "mixing_time", "coater_deviation", "speed", "temp", "calendering_deviation", "pressure", "electrode_maker_accuracy"]
en_var_list = ["viscosity", "density", "mixing_cost", "thickness", "coating_cost", "porosity", "tortuosity", "calendering_cost", "edge_quality", "optics", "cost"]

ex_matrix = np.array([
    [matrix_constants["environment-viscosity"], matrix_constants["mixing_time-viscosity"], 0, 0, 0, 0, 0, 0],
    [matrix_constants["environment-density"], 0, 0, 0, 0, 0, 0, 0],
    [0, matrix_constants["mixing_time-mixing_cost"], 0, 0, 0, 0, 0, 0],
    [0, 0, matrix_constants["coater_state-thickness"], matrix_constants["speed-thickness"], matrix_constants["temperature-thickness"], 0, 0, 0],
    [0, 0, 0, matrix_constants["speed-coating_cost"], matrix_constants["temp-coating_cost"], 0, 0, 0],
    [0, 0, 0, 0, 0, matrix_constants["calendering_state-porosity"], matrix_constants["pressure-porosity"], 0],
    [0, 0, 0, 0, 0, matrix_constants["calendering_state-tortuosity"], matrix_constants["pressure-tortuosity"], 0],
    [0, 0, 0, 0, 0, 0, matrix_constants["pressure-calendering_cost"], 0],
    [0, 0, 0, 0, 0, 0, 0, matrix_constants["electrode_maker-edge_quality"]],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0]
])

en_matrix = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [matrix_constants["viscosity-thickness"], matrix_constants["slurry_density-thickness"], 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, matrix_constants["thickness-porosity"], 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, matrix_constants["thickness-tortuosity"], 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, matrix_constants["porosity-edge_quality"], matrix_constants["tortuosity-edge_quality"], 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, matrix_constants["edge_quality-optics"], 0, 0],
    [0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0]
])

const_terms = np.array([matrix_constants["base_viscosity"], matrix_constants["base_slurry_density"], matrix_constants["base_mixing_cost"], matrix_constants["base_thickness"], matrix_constants["base_coating_cost"], matrix_constants["base_porosity"], matrix_constants["base_tortuosity"], matrix_constants["base_calendering_cost"], matrix_constants["base_edge_quality"], matrix_constants["base_optics"], 0])

battery_scm = causal_models.scm_from_matrices(ex_var_list, en_var_list, ex_matrix, en_matrix, const_terms)
