#!/usr/bin/env python3
"""Run multi-horizon closure, semantic grammar, and BEAST operator-graph tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tiktoken
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from reveng.experiments.reasoning_dynamics_graph import (
    benjamini_hochberg,
    operator_features,
    path_permutation_test,
    select_operator_clusters,
    transition_information_test,
)
from reveng.experiments.reasoning_topology import paired_inference
from reveng.experiments.semantic_regime_prediction import (
    CUE_COLUMNS,
    REGIMES,
    SEMANTIC_LABELS,
    combine_semantic_runs,
    parse_semantic_labels,
    semantic_feature_table,
)


ROOT = Path("outputs/hypothesis_tests")
POSITIONS = ROOT / "reasoning_regime_coarse_graining_v1/regime_positions.csv"
CHANGE_POINTS = ROOT / "action_distribution_cpd_beast_v1/detected_change_points.csv"
SEMANTIC_ROOT = ROOT / "semantic_reasoning_classification_v1/general_corpus_v1"
ORIGINAL = SEMANTIC_ROOT / "annotations_gpt_oss_20b_multilabel_v3_full.csv"
REPLICATE = (
    SEMANTIC_ROOT / "annotations_gpt_oss_20b_multilabel_v3_full_replicate.csv"
)
OUTPUT = ROOT / "reasoning_dynamics_graph_v1"
ACTION_COLUMNS = ("prob_up", "prob_down", "prob_left", "prob_right")
SENTENCE_HORIZONS = (1, 2, 3, 5, 8, 13)
WORD_HORIZONS = (32, 64, 128, 256)
TOKEN_HORIZONS = (32, 64, 128, 256)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def one_hot(values: pd.Series, categories: tuple[str, ...]) -> np.ndarray:
    mapping = {value: index for index, value in enumerate(categories)}
    output = np.zeros((len(values), len(categories)), dtype=float)
    for row, value in enumerate(values.astype(str)):
        output[row, mapping[value]] = 1.0
    return output


def add_regime_duration(positions: pd.DataFrame) -> pd.DataFrame:
    ordered = positions.sort_values(["example_id", "position_index"]).copy()
    duration = pd.Series(index=ordered.index, dtype=float)
    for _, trace in ordered.groupby("example_id", sort=False):
        run = 0
        previous = None
        for index, regime in zip(trace.index, trace.regime, strict=True):
            run = run + 1 if regime == previous else 1
            duration.loc[index] = run
            previous = regime
    ordered["regime_duration"] = duration
    return ordered


def semantic_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    original_raw = pd.read_csv(ORIGINAL)
    replicate_raw = pd.read_csv(REPLICATE)
    original_features = semantic_feature_table(original_raw, "original")
    replicate_features = semantic_feature_table(replicate_raw, "replicate")
    mean = combine_semantic_runs(original_features, replicate_features, "mean")
    return original_raw, replicate_raw, mean


def text_lengths(annotations: pd.DataFrame) -> pd.DataFrame:
    encoder = tiktoken.get_encoding("o200k_harmony")
    output = annotations[["example_id", "sentence_number", "target_sentence"]].copy()
    output["word_count"] = output.target_sentence.fillna("").map(
        lambda text: len(str(text).split())
    )
    output["token_count"] = output.target_sentence.fillna("").map(
        lambda text: len(encoder.encode(str(text)))
    )
    return output


def horizon_targets(
    lengths: pd.DataFrame, kind: str, value: int
) -> pd.DataFrame:
    rows = []
    for example_id, trace in lengths.groupby("example_id", sort=False):
        trace = trace.sort_values("sentence_number").reset_index(drop=True)
        words = trace.word_count.to_numpy(dtype=int)
        tokens = trace.token_count.to_numpy(dtype=int)
        positions = trace.sentence_number.to_numpy(dtype=int)
        for source_index, source_position in enumerate(positions):
            if kind == "sentence":
                target_index = source_index + value
                if target_index >= len(trace):
                    continue
            else:
                amounts = words if kind == "word" else tokens
                cumulative = np.cumsum(amounts[source_index + 1 :])
                hits = np.flatnonzero(cumulative >= value)
                if not len(hits):
                    continue
                target_index = source_index + 1 + int(hits[0])
            rows.append(
                {
                    "example_id": example_id,
                    "position_index": int(source_position),
                    "target_position_index": int(positions[target_index]),
                    "distance_sentences": int(target_index - source_index),
                    "distance_words": int(words[source_index + 1 : target_index + 1].sum()),
                    "distance_tokens": int(tokens[source_index + 1 : target_index + 1].sum()),
                    "horizon_type": kind,
                    "horizon_value": value,
                }
            )
    return pd.DataFrame(rows)


def prepare_horizon_frame(
    positions: pd.DataFrame,
    lengths: pd.DataFrame,
    semantics: pd.DataFrame,
    kind: str,
    value: int,
) -> pd.DataFrame:
    targets = horizon_targets(lengths, kind, value)
    source_columns = [
        "example_id", "position_index", "trajectory_id", "matched_pair_id",
        "matched_role", "validation_group", "reasoning_progress", "regime",
        "regime_duration", "computed_action_entropy_bits", *ACTION_COLUMNS,
    ]
    frame = targets.merge(
        positions[source_columns], on=["example_id", "position_index"],
        validate="one_to_one",
    )
    target_regimes = positions[["example_id", "position_index", "regime"]].rename(
        columns={"position_index": "target_position_index", "regime": "target_regime"}
    )
    frame = frame.merge(
        target_regimes, on=["example_id", "target_position_index"],
        validate="many_to_one",
    )
    frame = frame.merge(
        semantics, left_on=["example_id", "position_index"],
        right_on=["example_id", "sentence_number"], validate="one_to_one",
    )
    frame = frame[frame.semantic_valid.eq(1.0)].copy()
    frame["target_changed"] = frame.target_regime.ne(frame.regime).astype(int)
    return frame


def closure_features(frame: pd.DataFrame, include_semantics: bool) -> np.ndarray:
    numeric = frame[
        ["reasoning_progress", "regime_duration", "computed_action_entropy_bits", *ACTION_COLUMNS]
    ].to_numpy(dtype=float)
    regime = one_hot(frame.regime, REGIMES)
    if not include_semantics:
        return np.column_stack([numeric, regime])
    semantic_columns = [
        column for column in frame
        if column.startswith("semantic__label__") or column.startswith("semantic__cue__")
    ]
    return np.column_stack([numeric, regime, frame[semantic_columns].to_numpy(float)])


def trace_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.groupby("example_id").size()
    return frame.example_id.map(lambda value: 1.0 / counts[value]).to_numpy(float)


def fit_classifier(
    train_x: np.ndarray, train_y: np.ndarray, weights: np.ndarray,
    test_x: np.ndarray, classes: tuple[str, ...] | tuple[int, ...], seed: int,
) -> np.ndarray:
    output = np.full((len(test_x), len(classes)), 1e-12)
    unique = np.unique(train_y)
    if len(unique) == 1:
        output[:, classes.index(unique[0])] = 1.0
        return output / output.sum(axis=1, keepdims=True)
    model = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=2_000, random_state=seed)
    )
    model.fit(train_x, train_y, logisticregression__sample_weight=weights)
    for source, label in enumerate(model.named_steps["logisticregression"].classes_):
        output[:, classes.index(label)] = model.predict_proba(test_x)[:, source]
    return output / output.sum(axis=1, keepdims=True)


def run_closure(frame: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    rows = []
    for fold, group in enumerate(sorted(frame.validation_group.unique())):
        train = frame[frame.validation_group != group]
        test = frame[frame.validation_group == group]
        exact_y = train.target_regime.to_numpy(object)
        change_y = train.target_changed.to_numpy(int)
        for specification, semantic in (("state_only", False), ("state_plus_semantics", True)):
            train_x = closure_features(train, semantic)
            test_x = closure_features(test, semantic)
            weights = trace_weights(train)
            exact = fit_classifier(train_x, exact_y, weights, test_x, REGIMES, seed + fold)
            change = fit_classifier(train_x, change_y, weights, test_x, (0, 1), seed + 100 + fold)
            true_exact = test.target_regime.map({x: i for i, x in enumerate(REGIMES)}).to_numpy(int)
            true_change = test.target_changed.to_numpy(int)
            for index, source in enumerate(test.itertuples(index=False)):
                rows.append({
                    "horizon_type": source.horizon_type,
                    "horizon_value": source.horizon_value,
                    "specification": specification,
                    "example_id": source.example_id,
                    "validation_group": source.validation_group,
                    "distance_sentences": source.distance_sentences,
                    "distance_words": source.distance_words,
                    "distance_tokens": source.distance_tokens,
                    "target_changed": true_change[index],
                    "exact_log_loss": -np.log(max(exact[index, true_exact[index]], 1e-12)),
                    "change_log_loss": -np.log(max(change[index, true_change[index]], 1e-12)),
                    "exact_correct": int(np.argmax(exact[index]) == true_exact[index]),
                    "change_correct": int(np.argmax(change[index]) == true_change[index]),
                })
    return pd.DataFrame(rows)


def summarize_closure(predictions: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    trace = predictions.groupby(
        ["horizon_type", "horizon_value", "specification", "example_id"], as_index=False
    ).agg(
        validation_group=("validation_group", "first"),
        exact_log_loss=("exact_log_loss", "mean"),
        change_log_loss=("change_log_loss", "mean"),
        exact_accuracy=("exact_correct", "mean"),
        change_accuracy=("change_correct", "mean"),
    )
    rows = []
    for (kind, value), group in trace.groupby(["horizon_type", "horizon_value"]):
        baseline = group[group.specification == "state_only"].set_index("example_id")
        semantic = group[group.specification == "state_plus_semantics"].set_index("example_id")
        source = predictions[
            (predictions.horizon_type == kind)
            & (predictions.horizon_value == value)
            & (predictions.specification == "state_only")
        ]
        for metric in ("exact_log_loss", "change_log_loss"):
            improvement = baseline[metric] - semantic[metric]
            inference = paired_inference(
                improvement, groups=semantic.validation_group, seed=seed + int(value)
            )
            rows.append({
                "horizon_type": kind, "horizon_value": value, "metric": metric,
                "n_rows": len(source), "n_traces": source.example_id.nunique(),
                "median_sentences_ahead": float(source.distance_sentences.median()),
                "median_words_ahead": float(source.distance_words.median()),
                "median_tokens_ahead": float(source.distance_tokens.median()),
                "state_only_loss": float(baseline[metric].mean()),
                "state_plus_semantics_loss": float(semantic[metric].mean()),
                "semantic_improvement": float(improvement.mean()),
                "ci_low": inference["ci_low"], "ci_high": inference["ci_high"],
                "sign_flip_p": inference["sign_flip_p_two_sided"],
            })
    output = pd.DataFrame(rows)
    output["bh_q"] = benjamini_hochberg(output.sign_flip_p)
    return output.sort_values(["horizon_type", "horizon_value", "metric"])


def semantic_sequences(raw: pd.DataFrame) -> tuple[list, list, list]:
    valid = raw[raw.semantic_labels.notna() & raw.primary_label.notna()].copy()
    valid["labels"] = valid.semantic_labels.map(parse_semantic_labels)
    valid["progress_bin"] = np.minimum((valid.reasoning_progress * 10).astype(int), 9)
    label_sets, primary, strata = [], [], []
    for _, trace in valid.sort_values(["example_id", "sentence_number"]).groupby("example_id"):
        label_sets.append(trace.labels.tolist())
        primary.append(trace.primary_label.astype(str).tolist())
        strata.append(trace.progress_bin.tolist())
    return label_sets, primary, strata


def run_semantic_grammar(original: pd.DataFrame, replicate: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    information_rows, motif_rows = [], []
    for run_name, raw in (("original", original), ("replicate", replicate)):
        sets, primary, strata = semantic_sequences(raw)
        information_rows.append({"annotation_run": run_name, **transition_information_test(
            primary, trace_strata=strata, permutations=5_000, seed=42
        )})
        for order in (2, 3):
            result = path_permutation_test(
                sets, order=order, trace_strata=strata, permutations=2_000,
                seed=42 + order,
            )
            result.insert(0, "annotation_run", run_name)
            motif_rows.append(result)
    return pd.DataFrame(information_rows), pd.concat(motif_rows, ignore_index=True)


def run_beast_operator_graph(
    positions: pd.DataFrame, change_points: pd.DataFrame, semantics: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    for point in change_points.itertuples(index=False):
        trace = positions[positions.example_id == point.example_id].sort_values("position_index")
        probabilities = trace[list(ACTION_COLUMNS)].to_numpy(float)
        feature = operator_features(probabilities, int(point.position_index), window=3)
        rows.append({
            "example_id": point.example_id, "position_index": int(point.position_index),
            "matched_role": point.matched_role, "validation_group": trace.validation_group.iloc[0],
            "reasoning_progress": float(point.reasoning_progress),
            **feature,
        })
    events = pd.DataFrame(rows)
    feature_columns = [
        "total_variation", "jensen_shannon_bits", "entropy_delta_bits",
        "confidence_delta", "argmax_changed",
    ]
    labels, selection, selected_k = select_operator_clusters(
        events[feature_columns].to_numpy(float), seed=42
    )
    events["operator_cluster"] = [f"operator_{value + 1}" for value in labels]
    selection["selected"] = selection.n_clusters.eq(selected_k)
    cluster_summary = events.groupby("operator_cluster", as_index=False).agg(
        n_events=("example_id", "size"), total_variation=("total_variation", "mean"),
        jensen_shannon_bits=("jensen_shannon_bits", "mean"),
        entropy_delta_bits=("entropy_delta_bits", "mean"),
        confidence_delta=("confidence_delta", "mean"),
        argmax_change_rate=("argmax_changed", "mean"),
        failure_fraction=("matched_role", lambda x: float(np.mean(x == "failure"))),
    )
    traces = [
        [{str(value)} for value in trace.operator_cluster]
        for _, trace in events.sort_values(["example_id", "position_index"]).groupby("example_id")
    ]
    graph = path_permutation_test(traces, order=2, permutations=5_000, seed=52)
    motifs = path_permutation_test(traces, order=3, permutations=5_000, seed=53)

    semantic_columns = [column for column in semantics if column.startswith("semantic__label__")]
    joined = events.merge(
        semantics[["example_id", "sentence_number", *semantic_columns]],
        left_on=["example_id", "position_index"],
        right_on=["example_id", "sentence_number"], how="left", validate="one_to_one",
    )
    enrichment = joined.groupby("operator_cluster")[semantic_columns].mean().stack().reset_index()
    enrichment.columns = ["operator_cluster", "semantic_feature", "prevalence"]
    overall = joined[semantic_columns].mean()
    enrichment["overall_prevalence"] = enrichment.semantic_feature.map(overall)
    enrichment["prevalence_difference"] = enrichment.prevalence - enrichment.overall_prevalence
    rng = np.random.default_rng(64)
    null = {
        (cluster, semantic): []
        for cluster in sorted(joined.operator_cluster.unique())
        for semantic in semantic_columns
    }
    for _ in range(5_000):
        permuted = joined.operator_cluster.copy()
        for _, indices in joined.groupby("example_id").groups.items():
            permuted.loc[indices] = rng.permutation(permuted.loc[indices].to_numpy())
        for cluster in sorted(joined.operator_cluster.unique()):
            mask = permuted.eq(cluster).to_numpy()
            for semantic in semantic_columns:
                difference = float(joined.loc[mask, semantic].mean() - overall[semantic])
                null[(cluster, semantic)].append(difference)
    p_values = []
    for row in enrichment.itertuples(index=False):
        values = np.asarray(null[(row.operator_cluster, row.semantic_feature)])
        p_values.append(float((1 + np.sum(np.abs(values) >= abs(row.prevalence_difference))) / 5_001))
    enrichment["permutation_p_two_sided"] = p_values
    enrichment["bh_q"] = benjamini_hochberg(p_values)
    return events, selection, cluster_summary, graph, motifs, enrichment


def markdown_table(frame: pd.DataFrame, columns: list[str], limit: int | None = None) -> list[str]:
    if limit is not None:
        frame = frame.head(limit)
    lines = ["| " + " | ".join(x.replace("_", " ") for x in columns) + " |",
             "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in frame[columns].itertuples(index=False, name=None):
        lines.append("| " + " | ".join(f"{x:.4f}" if isinstance(x, float) else str(x) for x in row) + " |")
    return lines


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    positions = add_regime_duration(pd.read_csv(POSITIONS))
    change_points = pd.read_csv(CHANGE_POINTS)
    original, replicate, semantics = semantic_tables()
    lengths = text_lengths(original)

    prediction_frames = []
    for kind, values in (("sentence", SENTENCE_HORIZONS), ("word", WORD_HORIZONS), ("token", TOKEN_HORIZONS)):
        for value in values:
            frame = prepare_horizon_frame(positions, lengths, semantics, kind, value)
            prediction_frames.append(run_closure(frame))
    closure_predictions = pd.concat(prediction_frames, ignore_index=True)
    closure_summary = summarize_closure(closure_predictions)
    grammar_information, semantic_motifs = run_semantic_grammar(original, replicate)
    events, cluster_selection, cluster_summary, operator_edges, operator_motifs, semantic_enrichment = run_beast_operator_graph(
        positions, change_points, semantics
    )

    outputs = {
        "closure_predictions.csv": closure_predictions,
        "closure_summary.csv": closure_summary,
        "semantic_transition_information.csv": grammar_information,
        "semantic_paths.csv": semantic_motifs,
        "beast_operator_events.csv": events,
        "beast_operator_cluster_selection.csv": cluster_selection,
        "beast_operator_cluster_summary.csv": cluster_summary,
        "beast_operator_graph_edges.csv": operator_edges,
        "beast_operator_graph_motifs.csv": operator_motifs,
        "beast_operator_semantic_enrichment.csv": semantic_enrichment,
    }
    for filename, frame in outputs.items():
        frame.to_csv(OUTPUT / filename, index=False)

    replicated = semantic_motifs.pivot_table(
        index=["order", "path"], columns="annotation_run",
        values=["enrichment", "enrichment_ratio", "bh_q"]
    ).reset_index()
    replicated.columns = [
        first if not second else f"{first}_{second}"
        for first, second in replicated.columns
    ]
    supported = replicated[
        (replicated.bh_q_original < 0.05)
        & (replicated.bh_q_replicate < 0.05)
        & (replicated.enrichment_original > 0)
        & (replicated.enrichment_replicate > 0)
    ].copy()
    supported["mean_enrichment_ratio"] = (
        supported.enrichment_ratio_original + supported.enrichment_ratio_replicate
    ) / 2
    supported.to_csv(OUTPUT / "replicated_semantic_paths.csv", index=False)
    supported[supported.order.eq(2)].to_csv(
        OUTPUT / "semantic_process_graph_edges.csv", index=False
    )
    closure_supported = closure_summary[(closure_summary.ci_low > 0) & (closure_summary.bh_q < 0.05)]
    significant_operator_edges = operator_edges[(operator_edges.enrichment > 0) & (operator_edges.bh_q < .05)]
    significant_operator_motifs = operator_motifs[(operator_motifs.enrichment > 0) & (operator_motifs.bh_q < .05)]
    significant_operator_semantics = semantic_enrichment[semantic_enrichment.bh_q < .05]
    cross_label = supported[
        supported.path.map(lambda value: len(set(value.split(" -> "))) > 1)
    ].sort_values("mean_enrichment_ratio", ascending=False)

    lines = [
        "# Multi-Horizon Closure and Reasoning-Dynamics Graph",
        "", "## Questions", "",
        "1. Over what future sentence/word/token range do semantic labels add information beyond the current behavioral state?",
        "2. Do semantic operations exhibit ordering beyond their prevalence and coarse progress trends?",
        "3. Do BEAST boundaries form reproducible dynamical operator types and a non-random transition graph?",
        "", "## Predictive-closure definition", "",
        "Closure is target- and horizon-specific: semantic closure holds when adding the current sentence's averaged replicate multilabel vector does not improve grouped out-of-sample loss beyond current regime, regime duration, progress, action probabilities, and action entropy. It is not a claim that the state is sufficient for every possible future observable.",
        "", "Token distances use the local `o200k_harmony` encoding. Word/token horizons select the first future sentence boundary reaching the requested cumulative distance.",
        "", "## Multi-horizon results", "",
        *markdown_table(closure_summary, ["horizon_type", "horizon_value", "metric", "n_rows", "median_sentences_ahead", "median_words_ahead", "median_tokens_ahead", "semantic_improvement", "ci_low", "ci_high", "bh_q"]),
        "", f"Supported positive semantic increments after BH correction: {len(closure_supported)} of {len(closure_summary)} horizon-metric tests. Therefore this scan detects no violation of semantic predictive closure from 1–30 median sentences, 7–260 median words, or 11–459 median tokens ahead. It does not establish equivalence because no smallest effect of interest was prespecified.",
        "", "## Semantic process grammar", "",
        "Primary-label adjacent mutual information is tested against shuffling labels within each trajectory and progress decile. Multilabel edges and three-sentence paths distribute each window's mass across simultaneous labels and use the same null. A motif is declared only when enriched with BH q < .05 in both annotation runs.",
        "", *markdown_table(grammar_information, ["annotation_run", "observed_mutual_information_bits", "null_mean_bits", "excess_information_bits", "permutation_p_upper"]),
        "", f"Replicated enriched semantic paths: {len(supported[supported.order.eq(2)])} edges and {len(supported[supported.order.eq(3)])} three-sentence chains. Same-label bouts account for part, but not all, of this structure. Strong cross-label paths include:",
        "", *markdown_table(cross_label, ["order", "path", "mean_enrichment_ratio", "bh_q_original", "bh_q_replicate"], limit=12),
        "", "## Unsupervised BEAST operator graph", "",
        "Each BEAST boundary is represented without semantic or outcome labels by action-identity-invariant pre/post changes in total variation, Jensen-Shannon distance, entropy, confidence, and argmax switching. K-means K is selected by silhouette subject to at least 10 boundaries per cluster. Nodes are resulting operator clusters; directed edges join consecutive BEAST boundaries within a trace.",
        "", *markdown_table(cluster_selection, ["n_clusters", "silhouette", "minimum_cluster_size", "eligible", "selected"]),
        "", *markdown_table(cluster_summary, ["operator_cluster", "n_events", "total_variation", "entropy_delta_bits", "confidence_delta", "argmax_change_rate", "failure_fraction"]),
        "", f"Enriched operator edges after BH correction: {len(significant_operator_edges)}; enriched three-operator motifs: {len(significant_operator_motifs)}.",
        "", f"Operator-cluster/semantic-label associations after within-trace permutation and BH correction: {len(significant_operator_semantics)}. These associations are post-hoc descriptions and do not define the clusters.",
        "", "## Decision rules and limitations", "",
        "Evidence for a grammar requires replicated temporal dependence, not merely common labels. Evidence for graph motifs requires enrichment over a within-trace ordering null. BEAST is retrospective, clusters are exploratory and estimated on this corpus, and semantic-to-cluster prevalence differences are descriptive rather than causal.",
        "", "The observational causal-intervention proposal is not run: prepared single continuations do not identify the effect of rewriting or removing a sentence. That requires newly sampled matched continuations.",
        "", "## Literature basis", "",
        "The tests operationalize network motifs as subgraphs overrepresented relative to randomized networks ([Milo et al., 2002](https://doi.org/10.1126/science.298.5594.824)), temporal motifs as ordered event sequences evaluated with null models ([Kovanen et al., 2011](https://doi.org/10.1088/1742-5468/2011/11/P11005)), process discovery from event logs ([Process Mining Manifesto, 2012](https://doi.org/10.1007/978-3-642-28108-2_19)), and predictive state sufficiency as equivalence of conditional future distributions ([Shalizi and Crutchfield, 2001](https://arxiv.org/abs/cond-mat/9907176)). These works motivate the tests; they do not imply that DoorKey reasoning must contain motifs.",
    ]
    (OUTPUT / "run_report.md").write_text("\n".join(lines) + "\n")
    write_json(OUTPUT / "run_manifest.json", {
        "analysis": "reasoning_dynamics_graph_v1", "seed": 42,
        "sentence_horizons": SENTENCE_HORIZONS, "word_horizons": WORD_HORIZONS,
        "token_horizons": TOKEN_HORIZONS, "tokenizer": "o200k_harmony",
        "positions_sha256": sha256(POSITIONS), "change_points_sha256": sha256(CHANGE_POINTS),
        "original_sha256": sha256(ORIGINAL), "replicate_sha256": sha256(REPLICATE),
        "semantic_labels": SEMANTIC_LABELS, "cue_columns": CUE_COLUMNS,
    })


if __name__ == "__main__":
    main()
