# Progress and ETA

Each fit writes `status.json` and `history.jsonl` to its run directory and logs
lines like

```
[main round1] | 312/1200 shards | 26.1% rows | 18 workers (max 40) | 2.31M rows/s |
elapsed 0:42:10 | stage ETA 1:58:03 (1:47:12-2:09:40) | total ETA 2:31:20 -> Wed 09:14
```

`dfeols status <run dir or shard dir> --watch 60` shows it from any login node.
The ETA divides remaining work (from measured seconds per row) by the workers
connected now, never less than the largest shard still waiting, with a
bootstrap band; round 2 is projected from round 1's timing until measured.
