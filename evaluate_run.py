"""Check a trace and print the numbers for the report."""

from pathlib import Path

import argparse
import csv
import json
import math
from html import escape

from voice_kit import (
    read_trace,
    record_problems,
    summarize_stages,
    summarize_time_to_first_audio,
)
from voice_kit import stage_durations_milliseconds


WATERFALL_STAGES = ("asr", "llm_start", "speech_start")
OVERLAP_STAGE = "llm_remaining"

def percentile_waterfall(records, fraction):
    """Interpolate the same TTFA-ranked records for every component, preserving the sum."""
    ordered = sorted(records, key=lambda r: stage_durations_milliseconds(r)["time_to_first_audio"])
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    a, b = (stage_durations_milliseconds(ordered[i]) for i in (lower, upper))
    blend = lambda stage: a[stage] * (1 - weight) + b[stage] * weight
    components = {stage: blend(stage) for stage in WATERFALL_STAGES}
    return {
        "components_milliseconds": components,
        "ttfa_milliseconds": sum(components.values()),
        "llm_remaining_milliseconds": blend(OVERLAP_STAGE),
        "lower_turn_id": ordered[lower]["turn_id"],
        "upper_turn_id": ordered[upper]["turn_id"],
        "upper_weight": weight,
    }


def write_waterfall(path, waterfalls, count):
    """Write a standalone vector figure without adding a plotting dependency."""
    labels = ("ASR", "LLM start", "Speech start", "LLM remaining", "Total")
    colors = ("#2563eb", "#d97706", "#059669", "#fbbf24", "#374151")
    width, left, right, top = 1080, 150, 130, 80
    lane, bar_h, group_gap = 28, 20, 20
    tier_h, char_w = 14, 6.5  # event label row height; approx width of one 12px character
    group_h = len(labels) * lane
    halo = 'paint-order="stroke" stroke="white" stroke-width="3" stroke-linejoin="round"'

    # Per-percentile values: (name, asr, llm_start, speech_start, llm_remaining, ttfa)
    rows = []
    for name, wf in waterfalls.items():
        asr, llm, speech = (wf["components_milliseconds"][s] for s in WATERFALL_STAGES)
        rows.append((name, asr, llm, speech, wf["llm_remaining_milliseconds"], wf["ttfa_milliseconds"]))

    # Shared linear axis (1/2/5 x 10^n ticks) that also fits LLM completion, which can exceed TTFA.
    maximum = max(max(total, asr + llm + rem) for _, asr, llm, _, rem, total in rows) or 1
    raw = maximum / 6
    base = 10 ** math.floor(math.log10(raw))
    step = next(m * base for m in (1, 2, 5, 10) if m * base >= raw)
    ticks = [i * step for i in range(math.ceil(maximum / step) + 1)]
    scale = (width - left - right) / ticks[-1]

    def place_events(asr, llm, rem, total):
        """Return (ms, event, tier) per milestone. Overlapping labels stack, leftmost highest."""
        first = asr + llm
        events = sorted([(0, "End of speech"), (asr, "ASR final"), (first, "LLM first token"),
                         (first + rem, "LLM complete"), (total, "TTS first audio")])
        runs, end = [], -1e9
        for t, event in events:
            x = left + t * scale
            if x < end:  # overlaps the previous label, so join its run
                runs[-1].append((t, event))
                end = max(end, x + len(event) * char_w + 8)
            else:
                runs.append([(t, event)])
                end = x + len(event) * char_w + 8
        return [(t, event, len(run) - 1 - i) for run in runs for i, (t, event) in enumerate(run)]

    placed = [place_events(asr, llm, rem, total) for _, asr, llm, _, rem, total in rows]
    max_tier = max(tier for events in placed for _, _, tier in events)
    label_h = max_tier * tier_h + 22  # space above the first lane for event labels
    pitch = label_h + group_h + group_gap
    height = top + len(rows) * pitch + 60
    axis_y = height - 60

    def text(x, y, content, size=13, extra=""):
        return f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" {extra}>{content}</text>'

    body = []
    for row, (name, asr, llm, speech, rem, total) in enumerate(rows):
        gy = top + row * pitch + label_h  # y of the first lane
        body.append(text(30, gy + 18, escape(name), 18, 'font-weight="600"'))

        # Milestone lines and labels go first so bars and value labels draw on top of them.
        for t, event, tier in placed[row]:
            x = left + t * scale
            label_y = gy - 8 - tier * tier_h
            body.extend([
                f'<line x1="{x:.1f}" y1="{label_y + 3:.1f}" x2="{x:.1f}" y2="{gy + group_h}" '
                f'stroke="#6b7280" stroke-dasharray="3 3"/>',
                text(x + 3, label_y, event, 12, 'fill="#374151"'),
            ])

        first = asr + llm  # first spoken answer token
        # (start, duration, bold); LLM remaining starts at the first token and overlaps speech start.
        lanes = [(0, asr, False), (asr, llm, False), (first, speech, False),
                 (first, rem, False), (0, total, True)]
        for i, (start, value, bold) in enumerate(lanes):
            y = gy + i * lane
            x0 = left + start * scale
            x1 = x0 + max(value * scale, 1.5)  # zero-duration stages stay visible
            weight = 'font-weight="600"' if bold else ""
            share = f'<tspan fill="#9ca3af"> · {value / total:.0%}</tspan>' if total and i < 3 else ""
            body.extend([
                text(left - 10, y + 15, labels[i], extra=f'text-anchor="end" fill="#4b5563" {weight}'),
                f'<rect x="{x0:.2f}" y="{y}" width="{x1 - x0:.2f}" height="{bar_h}" fill="{colors[i]}" rx="2"/>',
                text(x1 + 6, y + 15, f"{value:,.1f} ms{share}", extra=f"{weight} {halo}"),
            ])

    grid = [f'<line x1="{left + t * scale:.1f}" y1="{top}" x2="{left + t * scale:.1f}" '
            f'y2="{axis_y}" stroke="#e5e7eb"/>' for t in ticks]
    grid += [text(left + t * scale, axis_y + 20, f"{t:,.0f}", 12, 'fill="#6b7280" text-anchor="middle"')
             for t in ticks]
    grid += [f'<line x1="{left}" y1="{axis_y}" x2="{left + ticks[-1] * scale:.1f}" y2="{axis_y}" stroke="#9ca3af"/>',
             text(left + ticks[-1] * scale / 2, axis_y + 40, "Milliseconds from end of user speech",
                  extra='fill="#6b7280" text-anchor="middle"')]

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="t d">',
        '<title id="t">Latency Waterfall Over 40 Turns</title>',
        f'<rect width="{width}" height="{height}" fill="white"/>',
        '<g font-family="Inter, Helvetica, Arial, sans-serif" fill="#111827">',
        text(width / 2, 34, "Latency Waterfall Over 40 Turns", 21, 'font-weight="600" text-anchor="middle"'),
        *grid, *body, "</g></svg>",
    ]
    Path(path).write_text("\n".join(parts), encoding="utf-8")
    

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, default=Path("run_output/trace.jsonl"))
    args = parser.parse_args(argv)
    
    records = read_trace(args.trace)
    for record in records:
        for problem in record_problems(record):
            # Nothing is printed when every turn passes.
            print(record["turn_id"], problem)

    print(summarize_time_to_first_audio(records))

    # p50 and p95 of each stage, for the table in your report.
    print(summarize_stages(records))

    waterfalls = {name: percentile_waterfall(records, fraction) for name, fraction in (("p50", 0.5), ("p95", 0.95))}
    summary = {
        "trace": str(args.trace.resolve()),
        "time_to_first_audio": summarize_time_to_first_audio(records),
        "stages": summarize_stages(records), 
        "waterfalls": waterfalls,
    }
    
    output = args.trace.parent / "evaluation"
    output.mkdir(parents=True, exist_ok=True)
    
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")            
    write_waterfall(output / "waterfall.svg", waterfalls, len(records))
    print(f"Saved summary.json and waterfall.svg in {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())