# Repository execution policy

## Test-time optimization is the default

For every test, evaluation, smoke run, and batch job, minimize wall-clock time while
preserving correctness, isolation, reproducibility, and API-rate-limit safety.

1. Run the smallest relevant test selection first; run the full regression suite once
   after the targeted checks pass.
2. Parallelize independent work by default. Use the documented worker defaults in
   `docs/TEST_EXECUTION_POLICY.md`; do not serialize independent API calls or test files.
3. Reuse safe caches, prepared indexes, fixtures, and completed output. Expensive jobs
   must support checkpoint/resume and incremental retry instead of restarting successes.
4. Separate deterministic/local tests from paid live API tests. Keep paid tests opt-in,
   use one representative smoke before a full batch, and batch independent requests.
5. Measure rather than assume: do not add workers when startup, shared-state contention,
   memory pressure, provider quotas, or retry storms make the run slower or unreliable.
6. Keep stateful, order-dependent, screenshot-update, index-mutation, and shared-file
   tests serial unless they have explicit isolation.
7. Record elapsed time and effective concurrency for long evaluation runs so the next
   run can choose better settings.

When adding a new long-running script, expose bounded concurrency and resumability
options (normally `--max-workers` and `--resume`) and choose a safe parallel default.
