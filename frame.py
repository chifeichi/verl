#!/usr/bin/env python3

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


MARKER = "[ROLLOUT_SAMPLE]"
ERROR_REASON = re.compile(r"error|timeout|failed|cancel", re.IGNORECASE)


def as_dict(value: object) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def nested_get(value: object, *path: str) -> object | None:
    current = value
    for key in path:
        current = as_dict(current).get(key)
    return current


def dataset_instance_id(row: pd.Series) -> str:
    candidates = (
        row.get("instance_id"),
        nested_get(row.get("extra_info"), "tools_kwargs", "reward", "metadata", "instance_id"),
        nested_get(row.get("extra_info"), "tools_kwargs", "task", "metadata", "instance_id"),
        nested_get(row.get("extra_info"), "instance_id"),
        nested_get(row.get("reward_model"), "ground_truth", "instance_id"),
    )
    for candidate in candidates:
        if candidate is not None and str(candidate):
            return str(candidate)
    return ""


def parse_logs(paths: list[str]) -> pd.DataFrame:
    records = []
    decoder = json.JSONDecoder()
    for path in paths:
        with open(path, encoding="utf-8", errors="replace") as stream:
            for line_number, line in enumerate(stream, 1):
                if MARKER not in line:
                    continue
                payload = line.split(MARKER, 1)[1].lstrip()
                try:
                    record, _ = decoder.raw_decode(payload)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid ROLLOUT_SAMPLE JSON at {path}:{line_number}: {exc}") from exc
                record["_log_path"] = path
                record["_log_line"] = line_number
                records.append(record)
    if not records:
        raise ValueError(f"No {MARKER} records found")
    frame = pd.DataFrame(records)
    required = {"instance_id", "num_turns", "prompt_tokens", "response_tokens", "model_tokens"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"ROLLOUT_SAMPLE records are missing fields: {sorted(missing)}")
    dedup = [name for name in ("global_steps", "session_id", "trajectory_index") if name in frame.columns]
    if dedup:
        frame = frame.drop_duplicates(dedup, keep="last")
    frame = frame[frame["instance_id"].fillna("").astype(str) != ""].copy()
    for name in ("num_turns", "prompt_tokens", "response_tokens", "model_tokens"):
        frame[name] = pd.to_numeric(frame[name], errors="coerce").fillna(0)
    frame["num_turns"] = frame["num_turns"].clip(lower=1)
    frame["non_model_tokens"] = (frame["response_tokens"] - frame["model_tokens"]).clip(lower=0)
    frame["model_tokens_per_turn"] = frame["model_tokens"] / frame["num_turns"]
    frame["decode_share"] = frame["model_tokens"] / frame["response_tokens"].clip(lower=1)
    frame["decode_to_context"] = frame["model_tokens"] / (
        frame["prompt_tokens"] + frame["non_model_tokens"]
    ).clip(lower=1)
    return frame


def bool_rate(group: pd.DataFrame, name: str) -> float:
    if name not in group:
        return 1.0
    values = group[name].dropna()
    if values.empty:
        return 1.0
    normalized = values.map(
        lambda value: value
        if isinstance(value, bool)
        else str(value).strip().lower() in {"1", "true", "yes"}
    )
    return float(normalized.mean())


def exit_success_rate(group: pd.DataFrame) -> float:
    if "claude_code_exit_code" not in group:
        return 1.0
    values = pd.to_numeric(group["claude_code_exit_code"], errors="coerce").dropna()
    return float((values == 0).mean()) if not values.empty else 1.0


def normal_reason_rate(group: pd.DataFrame) -> float:
    if "materialization_reason" not in group:
        return 1.0
    values = group["materialization_reason"].dropna().astype(str)
    return float((~values.str.contains(ERROR_REASON)).mean()) if not values.empty else 1.0


def summarize(records: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for instance_id, group in records.groupby("instance_id", sort=False):
        model_p50 = float(group["model_tokens"].median())
        model_p90 = float(group["model_tokens"].quantile(0.90))
        rows.append(
            {
                "instance_id": str(instance_id),
                "rollouts": len(group),
                "prompt_tokens_p50": float(group["prompt_tokens"].median()),
                "response_tokens_p50": float(group["response_tokens"].median()),
                "model_tokens_p25": float(group["model_tokens"].quantile(0.25)),
                "model_tokens_p50": model_p50,
                "model_tokens_p90": model_p90,
                "model_tokens_per_turn_p25": float(group["model_tokens_per_turn"].quantile(0.25)),
                "num_turns_p50": float(group["num_turns"].median()),
                "num_turns_p90": float(group["num_turns"].quantile(0.90)),
                "decode_share_p25": float(group["decode_share"].quantile(0.25)),
                "decode_to_context_p25": float(group["decode_to_context"].quantile(0.25)),
                "model_token_tail_ratio": model_p90 / max(model_p50, 1.0),
                "found_eval_status_rate": bool_rate(group, "found_eval_status"),
                "eval_completed_rate": bool_rate(group, "eval_completed"),
                "exit_success_rate": exit_success_rate(group),
                "normal_reason_rate": normal_reason_rate(group),
            }
        )
    return pd.DataFrame(rows)


def score(summary: pd.DataFrame, min_rollouts: int, min_valid_rate: float) -> pd.DataFrame:
    result = summary.copy()
    result["eligible"] = (
        (result["rollouts"] >= min_rollouts)
        & (result["model_tokens_p25"] > 0)
        & (result["found_eval_status_rate"] >= min_valid_rate)
        & (result["eval_completed_rate"] >= min_valid_rate)
        & (result["exit_success_rate"] >= min_valid_rate)
        & (result["normal_reason_rate"] >= min_valid_rate)
    )
    eligible = result["eligible"]
    if not eligible.any():
        raise ValueError("No eligible instances; lower --min-rollouts or --min-valid-rate")

    def percentile(column: str, ascending: bool = True) -> pd.Series:
        ranked = result.loc[eligible, column].rank(pct=True, ascending=ascending)
        return ranked.reindex(result.index).fillna(0.0)

    result["pd_score"] = (
        0.40 * percentile("model_tokens_per_turn_p25")
        + 0.20 * percentile("model_tokens_p25")
        + 0.15 * percentile("decode_share_p25")
        + 0.15 * percentile("decode_to_context_p25")
        + 0.05 * percentile("num_turns_p90", ascending=False)
        + 0.05 * percentile("model_token_tail_ratio", ascending=False)
    )
    result.loc[~eligible, "pd_score"] = np.nan
    result = result.sort_values(
        ["eligible", "pd_score", "model_tokens_per_turn_p25"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    result["rank"] = pd.Series(pd.NA, index=result.index, dtype="Int64")
    result.loc[result["eligible"], "rank"] = np.arange(1, int(result["eligible"].sum()) + 1)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", nargs="+", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="swe_rebench_pd64.parquet")
    parser.add_argument("--report", default="swe_rebench_pd_selection.csv")
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--min-rollouts", type=int, default=2)
    parser.add_argument("--min-valid-rate", type=float, default=0.75)
    args = parser.parse_args()

    records = parse_logs(args.log)
    summary = score(summarize(records), args.min_rollouts, args.min_valid_rate)
    selected_summary = summary[summary["eligible"]].head(args.num_samples).copy()
    if len(selected_summary) < args.num_samples:
        raise ValueError(
            f"Only {len(selected_summary)} eligible instances, fewer than --num-samples={args.num_samples}"
        )
    summary["selected"] = summary["rank"].isin(selected_summary["rank"])

    dataset = pd.read_parquet(args.input)
    ids = dataset.apply(dataset_instance_id, axis=1)
    duplicate_ids = ids[ids != ""].value_counts()
    duplicate_ids = duplicate_ids[duplicate_ids > 1]
    if not duplicate_ids.empty:
        raise ValueError(f"Input parquet has duplicate instance_id values: {duplicate_ids.index[:10].tolist()}")
    missing_ids = selected_summary.loc[~selected_summary["instance_id"].isin(set(ids)), "instance_id"].tolist()
    if missing_ids:
        raise ValueError(f"Selected instance_id values are missing from the input parquet: {missing_ids[:10]}")
    rank = dict(zip(selected_summary["instance_id"], selected_summary["rank"], strict=True))
    selected = dataset[ids.isin(rank)].copy()
    selected["_selection_rank"] = ids[ids.isin(rank)].map(rank).to_numpy()
    selected = selected.sort_values("_selection_rank").drop(columns="_selection_rank")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    selected.to_parquet(args.output, index=False)
    summary.to_csv(args.report, index=False)

    print(f"ROLLOUT_SAMPLE records: {len(records)}")
    print(f"Logged instances: {len(summary)}")
    print(f"Eligible instances: {int(summary['eligible'].sum())}")
    print(f"Selected instances: {len(selected_summary)}")
    print(f"Output parquet: {args.output}")
    print(f"Selection report: {args.report}")
    for row in selected_summary.itertuples(index=False):
        print(
            f"rank={row.rank} instance_id={row.instance_id} score={row.pd_score:.4f} "
            f"rollouts={row.rollouts} turns_p50={row.num_turns_p50:.1f} "
            f"prompt_p50={row.prompt_tokens_p50:.0f} response_p50={row.response_tokens_p50:.0f} "
            f"model_p50={row.model_tokens_p50:.0f} model_per_turn_p25={row.model_tokens_per_turn_p25:.1f} "
            f"decode_share_p25={row.decode_share_p25:.3f} valid_rate={row.found_eval_status_rate:.2f}"
        )


if __name__ == "__main__":
    main()


# python select_pd_benefit_samples.py \
#   --log 1.log \
#   --input swe_rebench_hard200.parquet \
#   --output swe_rebench_pd64.parquet \
#   --report swe_rebench_pd_selection.csv \
#   --num-samples 64 \
#   --min-rollouts 2 \
#   --min-valid-rate 0.75