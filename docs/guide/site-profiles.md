# Site profiles

```bash
dfeols config init          # writes ~/.config/dfeols/config.toml; edit [profiles.cluster]
dfeols config init --from-slurm --profile mysite --exclude 'interactive*'   # draft node classes
dfeols config show          # resolved profile
dfeols check                # start one worker through the scheduler, run a task
dfeols jobscript "python run.py"         # print the coordinator job script
dfeols submit run.py                     # submit run.py as the coordinator job
dfeols bench --profile cluster --rows 1e8
```

A profile holds everything site-specific; code never mentions a queue or a
module. Examples: [`examples/profiles/oscer.toml`](https://github.com/jemather/dfeols/blob/main/examples/profiles/oscer.toml)
(a tested Slurm site) and
[`examples/profiles/templates.toml`](https://github.com/jemather/dfeols/blob/main/examples/profiles/templates.toml) (PBS,
SGE, LSF, local; untested on real systems). Select one with
`dfeols.executor("name")`, `--profile name`, `$DFEOLS_PROFILE`, or `default`.
Walltimes are written once in any form (`12:00:00`, `2-00:00:00`) and
converted to each scheduler's native format.

On Slurm, `dfeols config init --from-slurm` drafts a profile's node classes
from `sinfo` and `scontrol show partition` (run on a login node, or from saved
output with `--sinfo`/`--scontrol`): partitions over the same nodes, limits
and policy collapse to one class, with cores (capped at `MaxCPUsPerNode`),
memory, walltime and nodes in service. GPU, down, hidden or redundant
partitions, and any matching `--exclude PATTERN`, are drafted with
`use = false` and the reason. It prints the draft; `--write` appends it to the
config and never replaces a profile.

Built-in site workarounds: on configless Slurm sites, a `/proc` `SLURM_CONF`
inside a job breaks `sbatch` and is cleared; the coordinator listens on a
private (cluster-internal) interface by default, because a campus-facing
address is often unreachable from compute nodes (`interface = "auto"`);
Polars' spill directory goes to scratch and its memory limit is pinned to the
job allocation.

Run the coordinator as a batch job (`dfeols submit`), not in an interactive
allocation: when the allocation expires, the coordinator and the fit go with it.
