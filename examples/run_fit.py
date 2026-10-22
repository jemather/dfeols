"""Template coordinator script for a real dataset. Submit with
    dfeols submit examples/run_fit.py --profile <your profile>

Edit SOURCE, the prep() call and SPECS. Nothing here is site-specific: the
scheduler, queue, modules and scratch path come from the profile.
"""

import dfeols

SOURCE = "/path/to/panel/*.parquet"

# Partition key: groups never split across shards. Choose the key that the
# most FEs (and ideally the cluster variable) are nested in. `dfeols info`
# on the prepared data shows what is local and what crosses shards.
PREP = dict(partition="unit",
            fixed_effects=["unit", "time"],      # every FE any spec uses
            clusters=["unit"],                   # every cluster variable any spec uses
            target_rows=2_000_000)

SPECS = {
    "main": dict(formula="y ~ x1 + x2 | unit + time", cluster="unit"),
    # "het": dict(formula="y ~ x1:i(cat) + x2 | unit + time", cluster="unit", weights="w"),
}


def main():
    prof = dfeols.profile()                       # $DFEOLS_PROFILE or the config default
    shard_dir = prof.scratch_dir("shards", "mydata")
    ex = dfeols.executor(prof)
    try:
        if (shard_dir / "dfeols_meta.json").exists():
            ds = dfeols.Dataset.open(shard_dir)
        else:
            # executor=ex: prep runs on the workers (Parquet input), so the
            # coordinator only holds the FE code tables
            ds = dfeols.prep(SOURCE, shard_dir=shard_dir, executor=ex, **PREP)
        print(ds)
        for label, spec in SPECS.items():
            wd = prof.scratch_dir("runs", label)
            res = dfeols.fit(ds, executor=ex, label=label, workdir=wd, **spec)
            print(res.summary())
            res.tidy().to_csv(wd / "coefs.csv")
    finally:
        ex.client.close()
        ex.cluster.close()


if __name__ == "__main__":
    main()
