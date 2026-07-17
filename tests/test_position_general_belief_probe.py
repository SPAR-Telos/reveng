from reveng.experiments.position_general_belief_probe import (
    _balanced_accuracy,
    _stratified_group_folds,
)


def test_stratified_group_folds_keep_each_state_in_one_fold() -> None:
    labels = {f"state_{index}": index % 2 for index in range(12)}
    folds = _stratified_group_folds(labels, n_folds=3, seed=7)

    assigned = [state for fold in folds for state in fold]
    assert sorted(assigned) == sorted(labels)
    assert len(assigned) == len(set(assigned))
    assert all({labels[state] for state in fold} == {0, 1} for fold in folds)


def test_balanced_accuracy_weights_classes_equally() -> None:
    labels = [0, 0, 0, 1]
    predictions = [0, 0, 0, 0]

    assert _balanced_accuracy(labels, predictions) == 0.5
