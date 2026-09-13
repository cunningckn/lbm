# Execution events

## Closed-loop controller port probe — 2026-09-14

The first baseline checkpoint completed all 50 official initial states, with
10 successes and no rollout exceptions. Before starting the next policy, the
controller's plain socket bind failed with `EADDRINUSE`. Inspection found no
listening process on port 8130. The probe did not enable address reuse, unlike
the HTTP server; recently closed connections can therefore reject that probe.

The controller now sets `SO_REUSEADDR` before probing, which still rejects an
active listener. The completed result is validated against the protocol and
training contract before reuse. The next arm had only its contract file and
had not run any trials; that directory was archived before retrying. The
original failure log, exit markers and dependent-queue failure logs are retained
on the server at `correctness/attempts/port-probe-failure`.

All three failed controllers had exited before they were restarted. The new
closed-loop log confirms reuse of the completed baseline and startup of the
candidate. All remaining arms subsequently completed, as did stability and mmap recovery.
This was an orchestration failure between arms, not a failed robot trial or a
reason to discard the completed baseline result. No other project's process
was terminated.
