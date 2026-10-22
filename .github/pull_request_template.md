**What this changes**

**Checks**
- [ ] `ruff check . && ruff format --check .`
- [ ] `pytest` passes (and `pytest --executor dask:2` if executors or prep changed)
- [ ] estimation changes keep agreement with pyfixest; new features have a comparison test
- [ ] `CHANGELOG.md` updated under *Unreleased*
