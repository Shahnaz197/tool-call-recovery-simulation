Simulates what happens when an agent's tool call times out and it has to choose between retrying, verifying or stopping.

Blog post: Retry, Verify or Stop? Simulating Recovery Policies for Failed Tool Calls

This is a simulation of recovery policies. It does not run an LLM or a real agent, and all rates and delays are assumptions.

A fake booking API can drop a request, or book the ticket and lose the response (an "ambiguous timeout"). Its status endpoint is eventually consistent, so a new booking may not be visible straight away. Five policies are compared:

naive: retry after every timeout (up to 3 retries)
verify: check status immediately, retry only if no booking is visible
verify_wait: wait 2 s, then check status
idempotency: retry with the same key, so the server ignores duplicates
stop: one attempt, one check, then escalate to a human

Each result is the mean of 20 replicates of 1,000 trials with 95% confidence intervals, and all policies share the same random draws. The script also checks that idempotency never duplicates and that the naive duplicate rate matches a closed-form calculation.
