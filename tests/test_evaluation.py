from pos.core import Fields


def test_evaluation(ds_lemma, tagger_evaluator, lemma_evaluator):
    accuracies, _ = tagger_evaluator(ds_lemma.get_field(Fields.GoldTags))
    print(accuracies)
    assert accuracies["Total"] == 1.0
    accuracies, _ = lemma_evaluator(
        ds_lemma.get_field(Fields.GoldLemmas),
    )
    print(accuracies)
    assert accuracies["Total"] == 1.0
