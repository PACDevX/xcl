"""Summarize a window of XCL's opt-in CPU presentation timeline."""

import argparse
import csv
import json
import math


def nearest_rank(values, fraction):
    return values[math.ceil(fraction * len(values)) - 1] if values else None


def summarize(rows, source, start_us, requested_end_us, stall_us):
    samples = []
    dropped = 0
    observed_end = 0
    for row in rows:
        at = int(row["time_us"])
        observed_end = max(observed_end, at)
        if row["row"] == "heartbeat":
            dropped = max(dropped, int(row["dropped_total"]))
        elif row["row"] == "present" and row["source"] == source:
            samples.append((int(row["sequence"]), at))
    end_us = min(observed_end, requested_end_us) if requested_end_us is not None else observed_end
    if end_us <= start_us:
        raise ValueError("the observed timeline does not cover the requested window")

    window = [(seq, at) for seq, at in samples if start_us <= at <= end_us]
    known = []
    gaps = 0
    for (before_seq, before), (after_seq, after) in zip(window, window[1:]):
        if after_seq == before_seq + 1 and after > before:
            known.append((after - before, before, after))
        else:
            gaps += 1

    durations = sorted(gap for gap, _, _ in known)
    tail = end_us - window[-1][1] if window else end_us - start_us
    slowest = durations[-math.ceil(len(durations) * 0.01):] if durations else []
    result = {
        "source": source,
        "start_us": start_us,
        "end_us": end_us,
        "requested_end_covered": requested_end_us is None or end_us == requested_end_us,
        "presents": len(window),
        "fps_including_stalls": len(window) * 1e6 / (end_us - start_us),
        "intervals": len(durations),
        "unknown_intervals": gaps,
        "dropped_total_in_recording": dropped,
        "min_ms": durations[0] / 1000 if durations else None,
        "p50_ms": nearest_rank(durations, 0.50) / 1000 if durations else None,
        "p95_ms": nearest_rank(durations, 0.95) / 1000 if durations else None,
        "one_percent_low_fps": len(slowest) * 1e6 / sum(slowest) if slowest else None,
        "stall_threshold_ms": stall_us / 1000,
        "stalls": [
            {"start_us": before, "end_us": after, "ms": gap / 1000}
            for gap, before, after in known if gap >= stall_us
        ],
        "end_without_present_ms": tail / 1000,
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("timeline", help="present-timing-*.csv")
    parser.add_argument("--source", choices=["presentx", "fallback"], default="presentx")
    parser.add_argument("--from", dest="start", type=float, default=0.0, help="log clock seconds")
    parser.add_argument("--to", dest="end", type=float, help="log clock seconds")
    parser.add_argument("--stall-ms", type=float, default=250.0)
    args = parser.parse_args()
    try:
        if args.start < 0 or args.stall_ms <= 0:
            raise ValueError("--from must be nonnegative and --stall-ms positive")
        with open(args.timeline, encoding="ascii", newline="") as file:
            result = summarize(
                csv.DictReader(file), args.source, round(args.start * 1e6),
                round(args.end * 1e6) if args.end is not None else None,
                round(args.stall_ms * 1000),
            )
    except (OSError, OverflowError, ValueError, KeyError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
