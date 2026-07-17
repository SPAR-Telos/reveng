# Table 1. Run Coverage and Event Counts

Summary of the eight DoorKey environment states used in the pilot. Sentence prefixes use canonical sentence boundaries; packed prefixes group nearby sentences. Action labels come from temperature 0.7 logprob queries over UP, DOWN, LEFT, and RIGHT. Commitments count states where the recommended action eventually matches the final full-trace action and remains stable. Belief probes count all behavioral belief-query rows.

| analysis unit | states | prefixes | action changes | optimal to suboptimal | sustained optimal to suboptimal | suboptimal to optimal | commitments | belief probes | activation-backed prefixes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Sentences | 8 | 662 | 61 | 12 | 3 | 15 | 8 | 8606 | 654 |
| Packed prefixes | 8 | 257 | 26 | 6 | 1 | 9 | 8 | 3341 | not used |
