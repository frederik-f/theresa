# Changelog

All notable changes are documented here. Versions follow
[Semantic Versioning](https://semver.org/).

## [1.0.0] — 2026-11-10

First release.

### Model

- 3D quarter unit cell of a Pollux-type canister in drift emplacement:
  canister, bentonite backfill, concrete liner, claystone host rock; Gmsh
  geometry and mesh, OpenGeoSys 6 `HEAT_CONDUCTION`.
- Run modes `steady` (`ψ` field), `kernel` (unit step response) and `direct`
  (decay-heat source, steady initial field).
- Reference decay-heat spectra DWR-Mix and CSD-V; custom exponential sums with
  any number of terms.

### Surrogate

- Closed-form Duhamel convolution of the step response with exponential-sum
  sources.
- Analytic treatment of surface temperature, geothermal gradient, depth,
  loading and interim storage.
- POD–GP surrogate over ten design variables (two spacings, conductivity and
  volumetric heat capacity of four materials), trained on 400 Latin-hypercube
  designs and included in the distribution.
- Interpolation uncertainty propagated to the temperature; warnings outside the
  trained range, for high uncertainty and for a maximum at the end of the
  record.
- Closed-form admissible loading for a temperature limit.
- Sobol sensitivity analysis (first-order and total indices, bootstrap
  standard errors).

### Interfaces

- Python API `theresa.predict`.
- Command line `theresa` with `predict`, `simulate`, `sobol`, `validate` and
  `train`.
- Streamlit GUI `theresa-gui`: German and English, light and dark theme,
  temperature and power charts, trained-range marks on the sliders, loading
  fitted to the design limit, direct OGS comparison with progress display and
  cancellation; prediction-only mode for hosting on Streamlit Community Cloud.

### Documentation

- API reference (Sphinx), theory (`docs/theory.pdf`), validation and
  performance (`docs/measurements.md`), model figures and an
  animated temperature history (`docs/figures`).
- Disclaimer in README, documentation, theory and GUI: hobby project,
  parameters not realistic, results not for design or safety assessments.

### Quality assurance

- pytest suite (209 tests) covering configuration, source term, convolution,
  geometry, mesh conversion, project files, runner, surrogate, sensitivity
  analysis, plots and GUI.
- Pre-commit hooks (ruff, whitespace, TOML/YAML) and GitHub Actions workflow:
  hooks, REUSE check, tests and documentation build on every push and pull
  request.

### Licensing

- Code under GPL-3.0-or-later, documentation and images under CC-BY-4.0;
  per-file assignment in `REUSE.toml` (REUSE 3.3).
