# Operational Epistemic States: Thresholds and Literature

## Purpose and status

This note defines how the DoorKey analyses should distinguish an expressed
`unknown` answer, diffuse uncertainty, a confident wrong answer, and an action
that fails despite an apparently correct belief. It is a design specification,
not a completed result.

Related repository notes:

- `docs/belief_transition_factorization.md` defines the action-transition
  outcomes and current behavioral-belief results.
- `docs/behavioral_probe_pathologies.md` documents prompting and logprob
  readout failures that must be excluded before epistemic interpretation.
- `docs/reasoning_belief_action_observational.md` documents the validation gate
  for representational probes.

The project-wide reusable BibTeX records are in `refs.bib`. The literature map
at the end of this note gives the relevant citation keys.

## Naming constraint

Use **probe-expressed ignorance**, not unqualified **model ignorance**, for an
`unknown` behavioral-probe response. A behavioral query measures what the
model is disposed to answer under a particular prompt and output format. A
representational probe measures what a trained readout can decode from a
chosen activation. Neither measurement alone establishes the model's literal
epistemic state.

Similarly, call a low-confidence activation readout **representationally
inconclusive**, not ignorant. It may reflect an uninformative representation,
an undertrained probe, or distribution shift at the reasoning position.

## Quantities retained for every behavioral belief query

For a categorical probe with normalized probabilities
`p_yes`, `p_no`, and `p_unknown`, and known environment truth `y` in
`{yes, no}`, retain:

| Quantity | Definition | Interpretation |
|---|---|---|
| `unknown_mass` | `p_unknown` | Propensity to give the explicit abstention answer |
| `factual_mass` | `p_yes + p_no` | Probability mass assigned to making a factual answer |
| `factual_choice` | `argmax(p_yes, p_no)` | Preferred factual answer, whether or not it is correct |
| `factual_commitment` | `max(p_yes, p_no)` | Absolute mass on the preferred factual answer |
| `conditional_factual_confidence` | `max(p_yes, p_no) / (p_yes + p_no)` | Yes/no confidence conditional on not answering `unknown` |
| `truth_mass` | `p_y` | Probability mass on the environment truth |
| `wrong_mass` | Probability of the other factual label | Probability mass on the false factual answer |
| `answer_entropy` | Entropy of the three-answer distribution | Dispersion diagnostic; not by itself ignorance |

Store the continuous quantities even when a categorical state is assigned.
The predictive analysis should use both the continuous measures and the
categories, so its conclusion does not depend entirely on one cutoff.

## Recommended primary decision rule

Apply this rule in order, separately for each belief query.

1. **Invalid measurement.** Required probabilities are missing, the surfaced
   answer and selected answer-token readout conflict, or another pathology in
   `docs/behavioral_probe_pathologies.md` applies. Do not convert this into an
   epistemic label.
2. **Probe-expressed ignorance.** `unknown` has strictly more probability mass
   than either `yes` or `no`. A tie is unresolved rather than ignorant.
3. **Unresolved factual belief.** A factual answer has the most mass, but its
   calibrated probability of being correct is below the commitment target.
4. **Committed correct belief.** A factual answer has the most mass, clears the
   commitment target, and matches the environment truth.
5. **Committed wrong belief.** A factual answer has the most mass, clears the
   commitment target, and contradicts the environment truth.

The first part of the rule is threshold-free: explicit ignorance means that
`unknown` is the model's most probable probe answer. The only fitted cutoff is
the reliability required before calling a factual answer committed.

For a direct sensitivity analysis of expressed ignorance, hold the commitment
rule fixed and compare three increasingly strict definitions:

- **top answer:** `p_unknown > max(p_yes, p_no)`;
- **majority mass:** `p_unknown >= 0.50`;
- **strong majority:** `p_unknown >= 0.67`.

The top-answer rule is primary because it asks whether the model would select
`unknown` from the three allowed answers without inventing a numerical cutoff.
The other two distinguish a weak preference for `unknown` from substantial
probability mass. Show the number of rows and trajectories assigned to
ignorance under each definition. Vary these definitions one at a time rather
than presenting a large grid of combined settings.

### Commitment target

Use an estimated factual-answer accuracy of **at least 80 percent** as the
primary operational definition of commitment. This is a proposed study choice,
not a literature-established universal boundary. Repeat the descriptive and
predictive results at 70 and 90 percent and show all three results plainly.

For a candidate factual-commitment cutoff `c`, define:

- `coverage(c)` as the proportion of eligible queries whose preferred factual
  answer has at least `c` probability mass;
- `selective_risk(c)` as the error rate among those covered queries.

The 80-percent reliability target is equivalently
`selective_risk(c) <= 0.20`. Choose the smallest cutoff that meets the target,
so the rule retains as many queries as possible at the declared reliability.

Estimate the mapping from raw `factual_commitment` to factual-answer accuracy
on calibration trajectories only. Keep all positions from one source
trajectory in the same split. Use a monotone calibrator when sample size
permits; otherwise use prespecified bins and trajectory-bootstrap intervals.
Pool belief questions only when individual question families are too sparse,
and report that pooling.

For a reliability target `r`, choose the smallest factual-commitment cutoff
whose estimated accuracy is at least `r`. A conservative version also requires
the lower end of its one-sided 95 percent trajectory-bootstrap interval to be
at least `r`. If no cutoff meets that requirement, that readout receives no
"committed" labels; it should not be rescued with a lower data-dependent
cutoff.

### Simple pilot rule before calibration

If a quick data audit is needed before the calibration pipeline exists, use:

- `unknown` is the strict top-probability answer: probe-expressed ignorance;
- otherwise `factual_commitment >= 0.80`: provisional factual commitment;
- otherwise: unresolved factual belief.

Mark these labels `provisional_threshold_0p80`. Do not use them as the final
inferential labels.

### Experimental labels versus a deployable monitor

The **correct** and **wrong** labels use the known DoorKey environment truth.
They are valid for controlled experimental analysis, but an online monitor
could assign them only if it also had an external source of state truth. A
monitor without that source can use expressed ignorance, unresolved belief,
confidence, entropy, and behavioral--representational disagreement, but it
cannot know from the model's outputs alone whether a confident belief is true.

## Threshold selection must be separated from action outcomes

Do not choose an ignorance or commitment cutoff because it predicts
recommendation change, optimality loss, or recovery well. That would make the
definition of the predictor depend on the outcome it is later claimed to
predict.

The order should be:

1. split by source trajectory;
2. repair or remove invalid behavioral readouts;
3. select and freeze thresholds using belief truth and calibration data only;
4. assign epistemic states to held-out trajectories;
5. test their associations with later action events.

For the sensitivity analysis, report the counts and action-event estimates at
each named cutoff. Avoid summaries such as "most settings agreed". State
directly whether the conclusion changes at 70, 80, or 90 percent factual
reliability, and whether it changes when ignorance requires a top answer, a
majority, or a strong majority. Change one rule at a time while holding the
other at its primary definition.

## Representational readouts

A representational belief probe must pass all of the following before it is
combined with the behavioral taxonomy:

1. evaluation is grouped by source trajectory or canonical grid state;
2. performance exceeds the class-balanced chance baseline on held-out groups;
3. the result is not explained by a matched control task or label leakage;
4. calibration is measured at the same reasoning positions used downstream;
5. performance and calibration are not confined to one reasoning-progress
   range;
6. results are shown across independently trained probes or bootstrap training
   sets.

After this gate, use **decodable correct**, **decodable wrong**, and
**representationally inconclusive**. Ensemble disagreement should be retained
as a separate measure of uncertainty about the readout. Do not treat the
softmax entropy of one linear probe as the model's epistemic uncertainty.

The released probes currently fail the intermediate-boundary transfer gate,
and the position-general pilot is not yet a validated measurement. Therefore,
the first analysis should be behavioral-first. Cross-channel states should be
added only after a new position-general probe passes this gate.

## Cross-channel and action states

Once a representational readout passes validation, derive the following states
without replacing the underlying continuous measurements:

| State | Behavioral readout | Representational readout | Action relation |
|---|---|---|---|
| Convergent correct belief | Committed correct | Decodable correct | Any |
| Convergent ignorance/inconclusiveness | Expressed ignorance or unresolved | Inconclusive | Any |
| Latent-information candidate | Ignorant, unresolved, or wrong | Decodable correct | Any |
| Probe-failure candidate | Committed correct | Wrong or inconclusive | Any |
| Convergent wrong belief | Committed wrong | Decodable wrong | Any |
| Belief-utilization failure | Committed correct, preferably convergent | Decodable correct if available | Recommended action contradicts the belief or is planner-suboptimal |

These names describe measurement patterns. Only an intervention that changes a
belief measure and then changes the action can support a causal belief claim.

## Primary tests

For optimality loss and recovery, compare models containing:

1. reasoning progress and current action confidence;
2. the continuous belief quantities above;
3. the categorical epistemic states;
4. both continuous quantities and states.

Keep full trajectories and matched states together in validation folds. The
structured states are useful only if they improve held-out discrimination or
calibration, or reveal a reproducible failure category that the continuous
scores conceal. Report category frequency, event rate, precision, coverage,
false-alarm rate, Brier score, log loss, and a discrimination measure suitable
for the event prevalence.

## Literature map

### Established distinctions and measurement methods

| Question | Relevant result | Use here | BibTeX key |
|---|---|---|---|
| Epistemic versus aleatoric uncertainty | Kendall and Gal distinguish reducible model uncertainty from irreducible data noise. | Do not interpret every diffuse distribution as ignorance. | `kendall_gal_2017_uncertainties` |
| Calibrated LM confidence | Jiang et al. find uncalibrated QA probabilities and study correction methods; Kadavath et al. find useful self-evaluation with incomplete cross-task calibration. | Calibrate the DoorKey readouts locally and test transfer. | `jiang_etal_2021_calibration`, `kadavath_etal_2022_know` |
| Explicit abstention | Cohen et al. train a dedicated `[IDK]` output and analyze its precision--recall tradeoff. | An explicit unknown answer is an action with its own calibration, not generic entropy. | `cohen_etal_2024_idk` |
| Semantic uncertainty | Kuhn et al. group meaning-equivalent generations before computing entropy. | Relevant if open-text belief answers replace constrained labels. | `kuhn_etal_2023_semantic_uncertainty` |
| Prediction sets | Angelopoulos and Bates explain distribution-free prediction sets and their coverage semantics. | A possible later representation of partial belief; guarantees require the relevant exchangeability conditions. | `angelopoulos_bates_2022_conformal` |
| Probe validity | Hewitt and Liang introduce control tasks; Pimentel et al. formalize probing in information-theoretic terms. | Distinguish information in a representation from information learned by the probe. | `hewitt_liang_2019_control_tasks`, `pimentel_etal_2020_information_probing` |
| Ensemble uncertainty | Lakshminarayanan et al. use independently trained predictors to obtain practical uncertainty estimates. | Use multiple probe fits to expose readout instability. | `lakshminarayanan_etal_2017_ensembles` |
| Evidential representations | Sensoy et al. propose Dirichlet evidence; Shen et al. show that the resulting epistemic interpretation can be unreliable. | Do not adopt a Dirichlet "ignorance mass" without validation and stress tests. | `sensoy_etal_2018_evidential`, `shen_etal_2024_mirage` |

### Project-specific interpretation

The literature establishes useful distinctions, calibration methods, explicit
abstention mechanisms, and probe controls. It does **not** establish that the
cutoffs in this note reveal a language model's literal knowledge. The
80-percent commitment target, the ordered decision rule, and the cross-channel
state names are our operational design choices and must be described as such.
