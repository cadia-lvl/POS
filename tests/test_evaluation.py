from pos.core import Fields


def test_evaluation(ds, tagger_evaluator, lemma_evaluator):
    accuracies, _ = tagger_evaluator(ds.get_field(Fields.GoldTags))
    print(accuracies)
    assert accuracies["Total"] == 1.0
    accuracies, _ = lemma_evaluator(
        ds.get_field(Fields.GoldLemmas),
    )
    print(accuracies)
    assert accuracies["Total"] == 1.0
