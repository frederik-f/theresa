# SPDX-License-Identifier: GPL-3.0-or-later
"""THERESA -- THErmal REpository Surrogate Analysis.

Peak temperature at the surface of a waste canister in drift emplacement over
1000 years, from a surrogate of a 3D OpenGeoSys 6 heat-conduction model.

Model
-----
A quarter of the unit cell around one canister, bounded by four symmetry
planes, i.e. an infinite array of simultaneously emplaced canisters. Four
materials: canister, backfill, liner, host rock. Fixed temperatures at ground
level and at the bottom boundary. See :mod:`theresa.geometry` and
:mod:`theresa.config`.

Method
------
The problem is linear, which separates three groups of inputs exactly:

1. Surface temperature and geothermal gradient enter through
   ``T_top + g * psi(x)`` (:mod:`theresa.steady_state`).
2. The decay-heat curve enters through a closed-form Duhamel convolution of the
   unit step response (:mod:`theresa.duhamel`); amplitudes, half-lives and the
   number of terms are free.
3. Depth shifts the temperature level through ``psi``.

The remaining dependence -- step response and ``psi`` on the clear canister
spacing, the drift spacing and the thermal properties of the four materials --
is interpolated by :mod:`theresa.surrogate` (POD and Gaussian processes).

Usage
-----
::

    theresa predict --canister-spacing 9.5 --a-drift 40 --plot T.png

::

    from theresa import predict
    answer = predict(canister_gap_m=9.5, a_drift_m=40.0, depth_m=700.0)
    print(answer.peak_celsius, answer.peak_time_years)

.. note::
   Geometry and material parameters are literature values for an illustrative
   study, not design data for a specific repository.

Modules
-------
- :mod:`theresa.config` -- parameters
- :mod:`theresa.spectra` -- reference decay-heat spectra
- :mod:`theresa.decay_heat` -- source terms
- :mod:`theresa.geometry` -- Gmsh geometry and mesh
- :mod:`theresa.mesh_conversion` -- Gmsh to OGS VTU
- :mod:`theresa.prj_builder` -- OGS project files
- :mod:`theresa.runner` -- OGS execution
- :mod:`theresa.steady_state` -- the ``psi`` field
- :mod:`theresa.kernel` -- step-response extraction
- :mod:`theresa.duhamel` -- convolution
- :mod:`theresa.pipeline` -- one design point end to end
- :mod:`theresa.doe` -- design of experiments and batch runs
- :mod:`theresa.surrogate` -- the interpolator
- :mod:`theresa.predict` -- prediction API
- :mod:`theresa.sobol` -- Sobol sensitivity analysis
- :mod:`theresa.plotting` -- result plots
- :mod:`theresa.cli` -- command-line interface
"""

from .predict import Prediction, predict

__all__ = ["Prediction", "predict"]
__version__ = "1.0.0"
