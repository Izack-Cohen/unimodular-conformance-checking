#!/usr/bin/env python3
\
\
\
\
\
\
\
\
\
\
\
   

import argparse
import sys
from pathlib import Path

import pm4py
import numpy as np
from pm4py.objects.log.obj import EventLog
from pm4py.objects.conversion.log import converter as log_converter


def get_trace_statistics(dataset_path):
                                                                      

    dataset_path = Path(dataset_path)
    dataset_name = dataset_path.stem

    print("=" * 80)
    print("TRACE LENGTH STATISTICS")
    print("=" * 80)
    print(f"\nDataset: {dataset_name}")
    print(f"Path: {dataset_path}")

                          
    if not dataset_path.exists():
        print(f"\n❌ ERROR: File not found: {dataset_path}")
        sys.exit(1)

                  
    print("\nLoading event log...")
    try:
        log = pm4py.read_xes(str(dataset_path))
    except Exception as e:
        print(f"\n❌ ERROR: Failed to load dataset: {e}")
        sys.exit(1)

                                                               
    if not isinstance(log, EventLog):
        print("Converting DataFrame to EventLog...")
        log = log_converter.apply(log, variant=log_converter.Variants.TO_EVENT_LOG)

    print(f"Loaded: {len(log)} traces")

                          
    print("\nCalculating trace lengths...")
    lengths = [len(trace) for trace in log]

    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    print(f"\nTotal traces: {len(lengths):,}")
    print(f"Total events: {sum(lengths):,}")

    print(f"\nTrace lengths:")
    print(f"  Min:    {min(lengths)} events")
    print(f"  Avg:    {sum(lengths) / len(lengths):.2f} events")
    print(f"  Median: {sorted(lengths)[len(lengths) // 2]} events")
    print(f"  Max:    {max(lengths)} events")
    print(f"  StdDev: {np.std(lengths):.2f} events")

    print(f"\nPercentiles:")
    print(f"  10th: {np.percentile(lengths, 10):.0f} events")
    print(f"  25th: {np.percentile(lengths, 25):.0f} events")
    print(f"  50th: {np.percentile(lengths, 50):.0f} events")
    print(f"  75th: {np.percentile(lengths, 75):.0f} events")
    print(f"  90th: {np.percentile(lengths, 90):.0f} events")
    print(f"  95th: {np.percentile(lengths, 95):.0f} events")
    print(f"  99th: {np.percentile(lengths, 99):.0f} events")

                            
    print("\n" + "=" * 80)
    print("DISTRIBUTION BY TRACE LENGTH")
    print("=" * 80)

    max_length = max(lengths)

                                  
    if max_length <= 50:
                                          
        ranges = [
            (0, 10, "Very Short (0-9)"),
            (10, 20, "Short (10-19)"),
            (20, 30, "Medium (20-29)"),
            (30, 50, "Long (30-49)"),
            (50, max_length + 1, "Very Long (50+)")
        ]
    elif max_length <= 100:
        ranges = [
            (0, 10, "Very Short (0-9)"),
            (10, 20, "Short (10-19)"),
            (20, 40, "Medium (20-39)"),
            (40, 60, "Long (40-59)"),
            (60, 100, "Very Long (60-99)"),
            (100, max_length + 1, "Extremely Long (100+)")
        ]
    else:
                             
        ranges = [
            (0, 10, "Very Short (0-9)"),
            (10, 20, "Short (10-19)"),
            (20, 40, "Medium (20-39)"),
            (40, 60, "Long (40-59)"),
            (60, 100, "Very Long (60-99)"),
            (100, 200, "Extremely Long (100-199)"),
            (200, max_length + 1, "Massive (200+)")
        ]

    print(f"\n{'Range':<25} {'Count':<10} {'Percentage':<12} {'Cumulative'}")
    print("-" * 80)

    cumulative = 0
    for min_len, max_len, label in ranges:
        count = sum(1 for l in lengths if min_len <= l < max_len)
        percentage = (count / len(lengths)) * 100
        cumulative += percentage

        if count > 0:                              
            print(f"{label:<25} {count:<10,} {percentage:>10.2f}%  {cumulative:>10.2f}%")

    print("\n" + "=" * 80)
    print("✅ Statistics calculated successfully!")
    print("\n" + "=" * 80)
    print("FOR YOUR PAPER:")
    print("=" * 80)

    mean_len = sum(lengths) / len(lengths)
    median_len = sorted(lengths)[len(lengths) // 2]

    print(f"\nDataset description:")
    print(f"  'The {dataset_name} dataset contains {len(lengths):,} process execution")
    print(f"  traces ranging from {min(lengths)} to {max(lengths)} events per trace")
    print(f"  (mean={mean_len:.1f}, median={median_len}).'")

    print(f"\nLaTeX table row:")
    print(
        f"  {dataset_name} & {len(lengths):,} & {min(lengths)} & {mean_len:.1f} & {median_len} & {max(lengths)} & {np.std(lengths):.1f} \\\\")

    print("\n" + "=" * 80)

    return {
        'dataset': dataset_name,
        'n_traces': len(lengths),
        'n_events': sum(lengths),
        'min': min(lengths),
        'max': max(lengths),
        'mean': mean_len,
        'median': median_len,
        'std': np.std(lengths),
        'percentiles': {
            10: np.percentile(lengths, 10),
            25: np.percentile(lengths, 25),
            50: np.percentile(lengths, 50),
            75: np.percentile(lengths, 75),
            90: np.percentile(lengths, 90),
            95: np.percentile(lengths, 95),
            99: np.percentile(lengths, 99)
        }
    }


def main():
    parser = argparse.ArgumentParser(
        description='Calculate trace length statistics for event log datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python dataset_statistics.py data/BPI_Challenge_2012.xes
  python dataset_statistics.py data/Sepsis_cases.xes
  python dataset_statistics.py data/Road_Traffic.xes
        """
    )

    parser.add_argument(
        'dataset',
        type=str,
        help='Path to XES event log file'
    )

    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Optional: Save statistics to JSON file'
    )

    args = parser.parse_args()

                          
    stats = get_trace_statistics(args.dataset)

                             
    if args.output:
        import json
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"\n💾 Statistics saved to: {output_path}")


if __name__ == '__main__':
    main()