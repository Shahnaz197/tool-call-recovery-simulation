import csv
import math
import random
import statistics
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")  # save figures to files; no window needed
import matplotlib.pyplot as plt



TRIALS_PER_REPLICATE = 1000
REPLICATES = 20
MAX_RETRIES = 3  # so at most MAX_RETRIES + 1 attempts
BASE_SEED = 42

P_NEVER_ARRIVES = 0.05  # request never reaches the server
AMBIGUOUS_RATES = [0.05, 0.20, 0.50]  # booking made, response lost

# How long until a new booking shows up in the status endpoint
VISIBILITY_DELAYS_MS = [0, 500, 1000, 2000, 3000, 5000]
DEFAULT_DELAY_MS = 2000  # used in the main comparison
FOCUS_RATE = 0.20  # used in the delay sweep and cost comparison

# Simulated timings (milliseconds)
REQUEST_TIMEOUT_MS = 1000
NORMAL_RESPONSE_MS = 200
PROCESSING_MS = 100
STATUS_CALL_MS = 100
VERIFY_WAIT_MS = 2000

STRATEGIES = ["naive", "verify", "verify_wait", "idempotency", "stop"]

NEVER, LOST, OK = "never", "lost", "ok"

# With these timings an *immediate* status check runs at
#   timeout + status_call - processing = 1000 ms after the booking was made,
# so immediate verification only sees the booking if VISIBILITY_DELAY_MS <= 1000.
# verify_wait sees it if VISIBILITY_DELAY_MS <= 3000.




class BookingAPI:
    def __init__(self, visibility_delay_ms):
        self.visibility_delay_ms = visibility_delay_ms
        self.bookings = []  # creation times (ms)
        self.idempotency_keys = set()
        self.clock = 0
        self.api_calls = 0

    def book_ticket(self, outcome, idempotency_key=None):
        """Returns 'success' or 'timeout' as the agent would see it."""
        self.api_calls += 1

        if outcome == NEVER:
            self.clock += REQUEST_TIMEOUT_MS
            return "timeout"

        # The request reached the server.
        already_seen = (
            idempotency_key is not None
            and idempotency_key in self.idempotency_keys
        )
        if not already_seen:
            self.bookings.append(self.clock + PROCESSING_MS)
            if idempotency_key is not None:
                self.idempotency_keys.add(idempotency_key)

        if outcome == LOST:
            self.clock += REQUEST_TIMEOUT_MS
            return "timeout"

        self.clock += NORMAL_RESPONSE_MS
        return "success"

    def visible_booking_count(self):
        """Status endpoint: eventually consistent view of bookings."""
        self.api_calls += 1
        self.clock += STATUS_CALL_MS
        return sum(
            1 for created in self.bookings
            if self.clock >= created + self.visibility_delay_ms
        )


# ---------------------------------------------------------------------
# Recovery policies
# ---------------------------------------------------------------------

def outcome_from_draw(u, p_lost):
    if u < P_NEVER_ARRIVES:
        return NEVER
    if u < P_NEVER_ARRIVES + p_lost:
        return LOST
    return OK


def run_trial(strategy, draws, p_lost, visibility_delay_ms):
    api = BookingAPI(visibility_delay_ms)
    key = "order-001" if strategy == "idempotency" else None
    max_attempts = 1 if strategy == "stop" else MAX_RETRIES + 1
    confirmed = False

    for attempt in range(max_attempts):
        outcome = outcome_from_draw(draws[attempt], p_lost)
        result = api.book_ticket(outcome, idempotency_key=key)

        if result == "success":
            confirmed = True
            break

        # The call timed out. What now?
        if strategy in ("verify", "verify_wait", "stop"):
            if strategy in ("verify_wait", "stop"):
                api.clock += VERIFY_WAIT_MS
            if api.visible_booking_count() > 0:
                confirmed = True  # evidence the booking exists
                break
        # naive / idempotency / verify-not-found: fall through and retry

    return len(api.bookings), confirmed, api.api_calls, api.clock



def mean_ci95(values):
    """Mean and 95% confidence half-width across replicates."""
    m = statistics.fmean(values)
    if len(values) < 2:
        return m, 0.0
    se = statistics.stdev(values) / math.sqrt(len(values))
    return m, 1.96 * se


def run_experiment():
    # Pre-generate the random draws once so every policy (and every
    # setting) is evaluated on exactly the same trials.
    draws_by_rep = []
    for rep in range(REPLICATES):
        rng = random.Random(BASE_SEED + rep)
        draws_by_rep.append([
            [rng.random() for _ in range(MAX_RETRIES + 1)]
            for _ in range(TRIALS_PER_REPLICATE)
        ])

    results = {}
    idempotency_duplicates_seen = 0

    for delay in VISIBILITY_DELAYS_MS:
        for rate in AMBIGUOUS_RATES:
            for strategy in STRATEGIES:
                per_rep = defaultdict(list)

                for draws_list in draws_by_rep:
                    c = defaultdict(float)
                    for draws in draws_list:
                        n_bookings, confirmed, calls, elapsed = run_trial(
                            strategy, draws, rate, delay
                        )
                        c["duplicate"] += n_bookings > 1
                        c["extra_bookings"] += max(0, n_bookings - 1)
                        c["no_booking"] += n_bookings == 0
                        c["clean_success"] += n_bookings == 1 and confirmed
                        c["escalated"] += not confirmed
                        c["booked_unconfirmed"] += n_bookings >= 1 and not confirmed
                        c["calls"] += calls
                        c["elapsed_ms"] += elapsed

                        if strategy == "idempotency" and n_bookings > 1:
                            idempotency_duplicates_seen += 1

                    n = TRIALS_PER_REPLICATE
                    per_rep["pct_duplicate"].append(100 * c["duplicate"] / n)
                    per_rep["avg_extra_bookings"].append(c["extra_bookings"] / n)
                    per_rep["pct_no_booking"].append(100 * c["no_booking"] / n)
                    per_rep["pct_clean_success"].append(100 * c["clean_success"] / n)
                    per_rep["pct_escalated"].append(100 * c["escalated"] / n)
                    per_rep["pct_booked_unconfirmed"].append(
                        100 * c["booked_unconfirmed"] / n
                    )
                    per_rep["avg_calls"].append(c["calls"] / n)
                    per_rep["avg_elapsed_ms"].append(c["elapsed_ms"] / n)

                results[(delay, rate, strategy)] = {
                    metric: mean_ci95(vals) for metric, vals in per_rep.items()
                }

    return results, idempotency_duplicates_seen




def naive_duplicate_probability(p_lost, attempts=MAX_RETRIES + 1):
    """Closed-form P(more than one booking) for the naive policy."""
    a, b = P_NEVER_ARRIVES, p_lost
    s = 1 - a - b
    running = {0: 1.0}  # bookings so far (capped at 2) -> probability
    finished = defaultdict(float)
    for _ in range(attempts):
        nxt = defaultdict(float)
        for k, pr in running.items():
            nxt[k] += pr * a  # never arrived: no booking, retry
            nxt[min(k + 1, 2)] += pr * b  # booked, response lost, retry
            finished[min(k + 1, 2)] += pr * s  # booked, got response, stop
        running = nxt
    for k, pr in running.items():  # retries exhausted
        finished[k] += pr
    return finished[2]


def run_sanity_checks(results, idempotency_duplicates_seen):
    print("\nSANITY CHECKS")
    print("-" * 70)

    assert idempotency_duplicates_seen == 0, "idempotency produced a duplicate!"
    print("OK  idempotency produced 0 duplicates in every trial.")

    for rate in AMBIGUOUS_RATES:
        expected = 100 * naive_duplicate_probability(rate)
        simulated, ci = results[(DEFAULT_DELAY_MS, rate, "naive")]["pct_duplicate"]
        diff = abs(expected - simulated)
        assert diff < 1.0, f"naive duplicate rate off at {rate:.0%}"
        print(
            f"OK  naive duplicates at {rate:.0%} ambiguous: "
            f"simulated {simulated:5.2f}% (+/-{ci:.2f}) vs exact {expected:5.2f}%"
        )

    # Naive never looks at the status endpoint, so the delay must not matter.
    base = results[(VISIBILITY_DELAYS_MS[0], FOCUS_RATE, "naive")]["pct_duplicate"][0]
    for delay in VISIBILITY_DELAYS_MS:
        other = results[(delay, FOCUS_RATE, "naive")]["pct_duplicate"][0]
        assert abs(base - other) < 1e-9
    print("OK  naive results do not depend on visibility delay (as expected).")



def print_table(results):
    print(f"\nRESULTS (visibility delay = {DEFAULT_DELAY_MS} ms; "
          f"mean over {REPLICATES} x {TRIALS_PER_REPLICATE} trials)")
    print("=" * 112)
    header = (
        f"{'ambig':>6} {'strategy':<12} {'dup %':>14} {'clean ok %':>14} "
        f"{'escalated %':>14} {'booked-but-unconfirmed %':>26} {'avg calls':>10}"
    )
    print(header)
    print("-" * 112)
    for rate in AMBIGUOUS_RATES:
        for strategy in STRATEGIES:
            r = results[(DEFAULT_DELAY_MS, rate, strategy)]
            fmt = lambda m: f"{r[m][0]:6.2f}+/-{r[m][1]:4.2f}"
            print(
                f"{rate:>6.0%} {strategy:<12} {fmt('pct_duplicate'):>14} "
                f"{fmt('pct_clean_success'):>14} {fmt('pct_escalated'):>14} "
                f"{fmt('pct_booked_unconfirmed'):>26} "
                f"{r['avg_calls'][0]:>10.2f}"
            )
        print("-" * 112)


def save_csv(results, path="agent_recovery_results_v2.csv"):
    rows = []
    for (delay, rate, strategy), metrics in results.items():
        row = {
            "visibility_delay_ms": delay,
            "ambiguous_timeout_rate": rate,
            "strategy": strategy,
        }
        for metric, (mean, ci) in metrics.items():
            row[metric] = round(mean, 4)
            row[metric + "_ci95"] = round(ci, 4)
        rows.append(row)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {path}")


def plot_duplicates_vs_timeout(results, path="duplicates_vs_timeout.png"):
    plt.figure(figsize=(7, 4.5))
    for strategy in STRATEGIES:
        xs = [r * 100 for r in AMBIGUOUS_RATES]
        ys = [results[(DEFAULT_DELAY_MS, r, strategy)]["pct_duplicate"][0]
              for r in AMBIGUOUS_RATES]
        es = [results[(DEFAULT_DELAY_MS, r, strategy)]["pct_duplicate"][1]
              for r in AMBIGUOUS_RATES]
        plt.errorbar(xs, ys, yerr=es, marker="o", capsize=3, label=strategy)
    plt.xlabel("Ambiguous timeout probability (%)")
    plt.ylabel("Trials with a duplicate booking (%)")
    plt.title(f"Duplicate bookings (status lag = {DEFAULT_DELAY_MS} ms)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def plot_duplicates_vs_visibility(results, path="duplicates_vs_visibility.png"):
    plt.figure(figsize=(7, 4.5))
    for strategy in ["naive", "verify", "verify_wait", "idempotency"]:
        ys = [results[(d, FOCUS_RATE, strategy)]["pct_duplicate"][0]
              for d in VISIBILITY_DELAYS_MS]
        es = [results[(d, FOCUS_RATE, strategy)]["pct_duplicate"][1]
              for d in VISIBILITY_DELAYS_MS]
        plt.errorbar(VISIBILITY_DELAYS_MS, ys, yerr=es, marker="o",
                     capsize=3, label=strategy)
    plt.xlabel("Status endpoint lag (ms)")
    plt.ylabel("Trials with a duplicate booking (%)")
    plt.title(f"Verify-first depends on status freshness "
              f"({FOCUS_RATE:.0%} ambiguous timeouts)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def plot_tradeoffs(results, path="tradeoffs.png"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    width = 0.25
    xs = range(len(STRATEGIES))

    def vals(metric):
        return [results[(DEFAULT_DELAY_MS, FOCUS_RATE, s)][metric][0]
                for s in STRATEGIES]

    ax1.bar([x - width for x in xs], vals("pct_clean_success"), width,
            label="clean success")
    ax1.bar(list(xs), vals("pct_duplicate"), width, label="duplicate")
    ax1.bar([x + width for x in xs], vals("pct_escalated"), width,
            label="escalated (unconfirmed)")
    ax1.set_xticks(list(xs))
    ax1.set_xticklabels(STRATEGIES)
    ax1.set_ylabel("% of trials")
    ax1.set_title("Outcomes")
    ax1.legend()

    ax2.bar(STRATEGIES, vals("avg_calls"), color="gray")
    ax2.set_ylabel("Average API calls per task")
    ax2.set_title("Cost")

    fig.suptitle(f"{FOCUS_RATE:.0%} ambiguous timeouts, "
                 f"status lag {DEFAULT_DELAY_MS} ms")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")




if __name__ == "__main__":
    results, idem_dupes = run_experiment()
    run_sanity_checks(results, idem_dupes)
    print_table(results)
    save_csv(results)
    plot_duplicates_vs_timeout(results)
    plot_duplicates_vs_visibility(results)
    plot_tradeoffs(results)