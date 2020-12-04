from pos.core import FieldedDataset, Fields, Vocab
from pos.evaluate import Experiment
from pos.cli import MORPHLEX_VOCAB_PATH, PRETRAINED_VOCAB_PATH


def test_evaluation(ds_lemma: FieldedDataset):
    evaluator = Experiment.all_accuracy_closure(
        ds_lemma,
        train_vocab=ds_lemma.get_vocab(),
        morphlex_vocab=Vocab.from_file(MORPHLEX_VOCAB_PATH),
        pretrained_vocab=Vocab.from_file(PRETRAINED_VOCAB_PATH),
    )
    accuracies, _ = evaluator(ds_lemma.get_field(Fields.GoldTags))
    print(accuracies)
    assert accuracies["Total"] == 1.0
