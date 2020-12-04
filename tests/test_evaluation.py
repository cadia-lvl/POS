from pos.core import FieldedDataset, Fields
from pos.evaluate import Experiment


def test_evaluation(ds_lemma: FieldedDataset, vocab_maps):
    evaluator = Experiment.all_accuracy_closure(ds_lemma, vocab_maps)
    accuracies, _ = evaluator(ds_lemma.get_field(Fields.GoldTags))
    print(accuracies)
    assert accuracies["Total"] == 1.0
