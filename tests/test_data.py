import torch
from pos.constants import Modules
from pos.core import Dicts, FieldedDataset, Fields, VocabMap
from pos.data import (
    BATCH_KEYS,
    bin_str_to_emb_pair,
    chunk_dataset,
    emb_pairs_to_dict,
    load_dicts,
    map_to_chars_and_index,
    map_to_chars_batch,
    map_to_index,
    read_datasets,
    wemb_str_to_emb_pair,
)
from pos.data.dataset import dechunk_dataset, get_adjusted_lengths
from pos.model.embeddings import TransformerEmbedding
from pos.utils import read_tsv, tokens_to_sentences
from torch import Tensor
from torch.utils.data.dataloader import DataLoader


def test_parse_bin_embedding():
    test = [
        "AA-deildina;[1,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]\n",
        "AA-deildin;[1,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]\n",
    ]
    embedding_dict = emb_pairs_to_dict(test, bin_str_to_emb_pair)
    # fmt: off
    assert embedding_dict["AA-deildina"].tolist() == [1,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    assert embedding_dict["AA-deildin"].tolist() == [1,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    # fmt: on


def test_parse_wemb():
    test = [
        "1340876 300\n",
        "</s> 0.075977 0.025685 -0.00068532 0.12494 -0.2065 0.11934 0.14417 0.31914 -0.047544 0.0031788 -0.28962 -0.1116 0.021398 0.11189 -0.27831 -0.12451 -0.20434 -0.0015718 -0.18203 -0.23465 -0.056694 0.015869 -0.019628 0.2944 0.031255 -0.082035 0.0067163 -0.1521 0.12566 -0.068878 -0.1358 0.011991 0.21512 0.0040974 -0.079886 0.15563 -0.084935 -0.18313 -0.106 -0.17579 0.19043 -0.022524 -0.10615 -0.22953 0.0342 0.1464 -0.1327 0.032771 -0.079249 -0.090165 0.31969 -0.030799 0.13153 -0.0060383 0.1774 -0.045092 -0.0064126 0.15505 0.1212 -0.015824 -0.14178 -0.10766 -0.061696 0.11542 -0.21847 -0.093152 0.14585 0.1008 -0.093933 0.35215 -0.0036202 0.096285 -0.049165 0.082816 0.12508 0.011616 0.21928 0.21695 -0.027946 -0.12758 -0.20246 0.12795 0.22385 -0.011815 -0.11849 -0.096164 -0.062516 0.2817 -0.13755 0.043147 0.037082 0.035357 -0.11647 -0.1418 0.18599 -0.0693 0.048342 0.067459 -0.07776 0.010091 -0.055684 0.0077467 -0.052856 0.035647 -0.25689 0.068266 0.15747 -0.19546 -0.017486 0.22853 0.057863 -0.0029496 0.15486 0.15011 0.098894 0.028168 -0.34968 0.10531 -0.14987 -0.021832 0.13879 0.00024128 0.062581 0.071624 -0.085334 0.050824 -0.10034 0.11906 -0.016801 0.15923 0.031911 0.0055539 -0.084445 -0.083484 0.067278 0.075715 -0.02786 -0.11039 0.24014 -0.017517 -0.021569 0.23831 -0.12698 0.0089676 0.17419 -0.072193 -0.11364 0.061876 -0.093116 -0.36842 0.050712 0.023042 -0.028991 0.09209 -0.12536 0.17972 0.19026 -0.0095842 0.071792 -0.030506 0.038806 -0.082041 -0.15154 0.069356 -0.048743 0.077301 -0.13943 -0.084722 -0.012721 0.045409 -0.087039 -0.14601 -0.068261 0.02851 -0.14332 0.13505 -0.20354 -0.0058208 0.10343 0.38409 -0.13406 -0.011208 0.054665 -0.032421 -0.32708 -0.011012 0.024211 0.077976 0.0040071 0.11926 0.095916 -0.03051 -0.09197 -0.033205 -0.13504 0.034082 -0.19821 -0.074038 -0.17653 0.079749 0.12565 -0.25496 -0.20914 0.048301 0.07815 0.12358 -0.016316 -0.030747 -0.14587 -0.14909 0.10574 0.049295 0.095624 -0.035668 0.022761 -0.10569 0.11628 0.088412 0.015155 -0.23346 -0.051093 0.01723 -0.058168 -0.17661 0.083597 0.1091 0.074263 -0.14775 -0.15083 -0.071676 -0.078866 -0.17248 -0.09665 -0.071646 -0.15948 0.098364 0.0043641 0.04245 -0.25275 -0.0096076 -0.17701 0.26893 -0.37961 -0.088137 -0.056866 0.096092 -0.058289 -0.1417 0.10774 0.0878 -0.13317 0.10694 0.14916 0.03528 -0.20491 0.050946 0.12683 0.080925 -0.084883 0.040907 0.045555 0.019904 0.15942 -0.061566 0.02459 0.26197 -0.042496 -0.13827 -0.14195 0.10312 0.23241 0.18791 0.45344 0.0619 0.17062 0.1181 -0.11684 0.049233 -0.18428 -0.2276 0.18566 -0.16241 0.06044 0.16506 -0.15652 0.077407 -0.088054 -0.07865 -0.25065 0.0048217 -0.10898 -0.035774 0.042697 0.049362 0.03642 0.067454 -0.055721 -0.041073 -0.20843 0.10055\n",
        ". 0.05431 0.0024818 0.0010406 0.10481 -0.1578 0.069616 0.1663 0.30658 -0.052971 -0.012768 -0.30543 -0.078391 0.016214 0.12461 -0.27869 -0.12881 -0.21276 -0.025674 -0.21862 -0.20527 -0.02308 0.049061 0.026404 0.30642 0.030449 -0.11721 -0.011167 -0.12117 0.12133 -0.068081 -0.16409 -0.028847 0.25494 -0.025145 -0.1233 0.16653 -0.063059 -0.13018 -0.13948 -0.17903 0.22837 0.014854 -0.095656 -0.18372 0.06622 0.11177 -0.14605 0.014314 -0.04304 -0.084607 0.33001 -0.039672 0.10694 -0.029362 0.18235 -0.053034 -0.043924 0.12657 0.10586 0.0089735 -0.17547 -0.091721 -0.060285 0.11344 -0.24863 -0.070779 0.12871 0.11449 -0.048764 0.33103 -0.0028953 0.090834 -0.056963 0.064784 0.13121 -0.027809 0.25618 0.20524 -0.030847 -0.10806 -0.25367 0.14877 0.20009 -0.012711 -0.10768 -0.10025 -0.040142 0.30946 -0.12392 0.058069 0.0014803 0.0052063 -0.096491 -0.10139 0.18756 -0.057492 -0.0006844 0.05056 -0.071322 0.015029 -0.062826 -0.0033695 -0.042992 0.029543 -0.27178 0.056602 0.1195 -0.20762 0.00733 0.26135 0.070992 0.0031749 0.14105 0.13394 0.064886 0.009939 -0.29671 0.16506 -0.12723 -0.0059324 0.18847 0.0062121 0.085697 0.037486 -0.083877 0.091125 -0.086268 0.16143 -0.05226 0.15437 0.050279 4.7654e-05 -0.060292 -0.088363 0.059568 0.061832 -0.080757 -0.14316 0.21216 -0.015359 -0.0033211 0.21352 -0.15734 0.0056357 0.20625 -0.090438 -0.13685 0.059074 -0.045881 -0.37346 0.06181 -0.017948 -0.020045 0.12032 -0.13348 0.14216 0.1844 -0.017722 0.038104 -0.012497 0.047691 -0.084907 -0.15381 0.070729 -0.081926 0.10532 -0.1208 -0.047206 0.0035534 0.052811 -0.081044 -0.14739 -0.086468 0.023295 -0.16176 0.14592 -0.20558 0.01939 0.095705 0.40724 -0.1258 -0.014661 0.076479 -0.026861 -0.28186 0.03608 0.018838 0.092869 0.024971 0.17152 0.075297 -0.028556 -0.098291 -0.0041797 -0.15909 0.0078698 -0.17718 -0.070422 -0.18837 0.06797 0.12163 -0.21266 -0.19437 0.024254 0.08154 0.11797 -0.014957 -0.061535 -0.11435 -0.15158 0.073681 0.077131 0.055916 -0.04628 0.028507 -0.060497 0.08308 0.11859 -0.0070604 -0.22775 -0.015577 0.022559 -0.046226 -0.22118 0.076958 0.094774 0.077995 -0.20287 -0.15239 -0.048023 -0.050555 -0.16429 -0.10967 -0.022875 -0.14339 0.090035 0.0038961 0.028857 -0.20737 0.027809 -0.17458 0.26268 -0.33302 -0.090267 -0.075347 0.07355 -0.0059844 -0.15754 0.071119 0.083696 -0.11894 0.10315 0.14336 0.050512 -0.17846 0.059857 0.17208 0.082009 -0.11091 0.079476 0.048901 0.049977 0.14018 -0.054505 0.008947 0.24975 -0.027567 -0.14832 -0.11974 0.13201 0.28712 0.17284 0.46652 0.099915 0.14558 0.11091 -0.12942 0.05459 -0.1832 -0.23799 0.18598 -0.12863 0.057736 0.1735 -0.16933 0.074836 -0.089106 -0.091376 -0.2451 0.0050707 -0.11385 -0.064311 -0.00033057 0.062836 0.032948 0.060489 -0.049488 -0.027227 -0.218 0.094666\n",
    ]
    embedding_dict = emb_pairs_to_dict(test, wemb_str_to_emb_pair)
    # fmt: off
    assert torch.allclose(embedding_dict["</s>"], Tensor([0.075977,0.025685,-0.00068532,0.12494,-0.2065,0.11934,0.14417,0.31914,-0.047544,0.0031788,-0.28962,-0.1116,0.021398,0.11189,-0.27831,-0.12451,-0.20434,-0.0015718,-0.18203,-0.23465,-0.056694,0.015869,-0.019628,0.2944,0.031255,-0.082035,0.0067163,-0.1521,0.12566,-0.068878,-0.1358,0.011991,0.21512,0.0040974,-0.079886,0.15563,-0.084935,-0.18313,-0.106,-0.17579,0.19043,-0.022524,-0.10615,-0.22953,0.0342,0.1464,-0.1327,0.032771,-0.079249,-0.090165,0.31969,-0.030799,0.13153,-0.0060383,0.1774,-0.045092,-0.0064126,0.15505,0.1212,-0.015824,-0.14178,-0.10766,-0.061696,0.11542,-0.21847,-0.093152,0.14585,0.1008,-0.093933,0.35215,-0.0036202,0.096285,-0.049165,0.082816,0.12508,0.011616,0.21928,0.21695,-0.027946,-0.12758,-0.20246,0.12795,0.22385,-0.011815,-0.11849,-0.096164,-0.062516,0.2817,-0.13755,0.043147,0.037082,0.035357,-0.11647,-0.1418,0.18599,-0.0693,0.048342,0.067459,-0.07776,0.010091,-0.055684,0.0077467,-0.052856,0.035647,-0.25689,0.068266,0.15747,-0.19546,-0.017486,0.22853,0.057863,-0.0029496,0.15486,0.15011,0.098894,0.028168,-0.34968,0.10531,-0.14987,-0.021832,0.13879,0.00024128,0.062581,0.071624,-0.085334,0.050824,-0.10034,0.11906,-0.016801,0.15923,0.031911,0.0055539,-0.084445,-0.083484,0.067278,0.075715,-0.02786,-0.11039,0.24014,-0.017517,-0.021569,0.23831,-0.12698,0.0089676,0.17419,-0.072193,-0.11364,0.061876,-0.093116,-0.36842,0.050712,0.023042,-0.028991,0.09209,-0.12536,0.17972,0.19026,-0.0095842,0.071792,-0.030506,0.038806,-0.082041,-0.15154,0.069356,-0.048743,0.077301,-0.13943,-0.084722,-0.012721,0.045409,-0.087039,-0.14601,-0.068261,0.02851,-0.14332,0.13505,-0.20354,-0.0058208,0.10343,0.38409,-0.13406,-0.011208,0.054665,-0.032421,-0.32708,-0.011012,0.024211,0.077976,0.0040071,0.11926,0.095916,-0.03051,-0.09197,-0.033205,-0.13504,0.034082,-0.19821,-0.074038,-0.17653,0.079749,0.12565,-0.25496,-0.20914,0.048301,0.07815,0.12358,-0.016316,-0.030747,-0.14587,-0.14909,0.10574,0.049295,0.095624,-0.035668,0.022761,-0.10569,0.11628,0.088412,0.015155,-0.23346,-0.051093,0.01723,-0.058168,-0.17661,0.083597,0.1091,0.074263,-0.14775,-0.15083,-0.071676,-0.078866,-0.17248,-0.09665,-0.071646,-0.15948,0.098364,0.0043641,0.04245,-0.25275,-0.0096076,-0.17701,0.26893,-0.37961,-0.088137,-0.056866,0.096092,-0.058289,-0.1417,0.10774,0.0878,-0.13317,0.10694,0.14916,0.03528,-0.20491,0.050946,0.12683,0.080925,-0.084883,0.040907,0.045555,0.019904,0.15942,-0.061566,0.02459,0.26197,-0.042496,-0.13827,-0.14195,0.10312,0.23241,0.18791,0.45344,0.0619,0.17062,0.1181,-0.11684,0.049233,-0.18428,-0.2276,0.18566,-0.16241,0.06044,0.16506,-0.15652,0.077407,-0.088054,-0.07865,-0.25065,0.0048217,-0.10898,-0.035774,0.042697,0.049362,0.03642,0.067454,-0.055721,-0.041073,-0.20843,0.10055]), atol=1e-04)
    assert torch.allclose(embedding_dict["."], Tensor([0.05431,0.0024818,0.0010406,0.10481,-0.1578,0.069616,0.1663,0.30658,-0.052971,-0.012768,-0.30543,-0.078391,0.016214,0.12461,-0.27869,-0.12881,-0.21276,-0.025674,-0.21862,-0.20527,-0.02308,0.049061,0.026404,0.30642,0.030449,-0.11721,-0.011167,-0.12117,0.12133,-0.068081,-0.16409,-0.028847,0.25494,-0.025145,-0.1233,0.16653,-0.063059,-0.13018,-0.13948,-0.17903,0.22837,0.014854,-0.095656,-0.18372,0.06622,0.11177,-0.14605,0.014314,-0.04304,-0.084607,0.33001,-0.039672,0.10694,-0.029362,0.18235,-0.053034,-0.043924,0.12657,0.10586,0.0089735,-0.17547,-0.091721,-0.060285,0.11344,-0.24863,-0.070779,0.12871,0.11449,-0.048764,0.33103,-0.0028953,0.090834,-0.056963,0.064784,0.13121,-0.027809,0.25618,0.20524,-0.030847,-0.10806,-0.25367,0.14877,0.20009,-0.012711,-0.10768,-0.10025,-0.040142,0.30946,-0.12392,0.058069,0.0014803,0.0052063,-0.096491,-0.10139,0.18756,-0.057492,-0.0006844,0.05056,-0.071322,0.015029,-0.062826,-0.0033695,-0.042992,0.029543,-0.27178,0.056602,0.1195,-0.20762,0.00733,0.26135,0.070992,0.0031749,0.14105,0.13394,0.064886,0.009939,-0.29671,0.16506,-0.12723,-0.0059324,0.18847,0.0062121,0.085697,0.037486,-0.083877,0.091125,-0.086268,0.16143,-0.05226,0.15437,0.050279,4.7654e-05,-0.060292,-0.088363,0.059568,0.061832,-0.080757,-0.14316,0.21216,-0.015359,-0.0033211,0.21352,-0.15734,0.0056357,0.20625,-0.090438,-0.13685,0.059074,-0.045881,-0.37346,0.06181,-0.017948,-0.020045,0.12032,-0.13348,0.14216,0.1844,-0.017722,0.038104,-0.012497,0.047691,-0.084907,-0.15381,0.070729,-0.081926,0.10532,-0.1208,-0.047206,0.0035534,0.052811,-0.081044,-0.14739,-0.086468,0.023295,-0.16176,0.14592,-0.20558,0.01939,0.095705,0.40724,-0.1258,-0.014661,0.076479,-0.026861,-0.28186,0.03608,0.018838,0.092869,0.024971,0.17152,0.075297,-0.028556,-0.098291,-0.0041797,-0.15909,0.0078698,-0.17718,-0.070422,-0.18837,0.06797,0.12163,-0.21266,-0.19437,0.024254,0.08154,0.11797,-0.014957,-0.061535,-0.11435,-0.15158,0.073681,0.077131,0.055916,-0.04628,0.028507,-0.060497,0.08308,0.11859,-0.0070604,-0.22775,-0.015577,0.022559,-0.046226,-0.22118,0.076958,0.094774,0.077995,-0.20287,-0.15239,-0.048023,-0.050555,-0.16429,-0.10967,-0.022875,-0.14339,0.090035,0.0038961,0.028857,-0.20737,0.027809,-0.17458,0.26268,-0.33302,-0.090267,-0.075347,0.07355,-0.0059844,-0.15754,0.071119,0.083696,-0.11894,0.10315,0.14336,0.050512,-0.17846,0.059857,0.17208,0.082009,-0.11091,0.079476,0.048901,0.049977,0.14018,-0.054505,0.008947,0.24975,-0.027567,-0.14832,-0.11974,0.13201,0.28712,0.17284,0.46652,0.099915,0.14558,0.11091,-0.12942,0.05459,-0.1832,-0.23799,0.18598,-0.12863,0.057736,0.1735,-0.16933,0.074836,-0.089106,-0.091376,-0.2451,0.0050707,-0.11385,-0.064311,-0.00033057,0.062836,0.032948,0.060489,-0.049488,-0.027227,-0.218,0.094666]), atol=1e-04)
    # fmt: on


def test_read_tsv(test_tsv_lemma_file):
    with open(test_tsv_lemma_file) as f:
        tmp = list(tokens_to_sentences(read_tsv(f)))
        print(tmp)
        tmp == [
            (("Hæ",), ("a",), ("hæ",)),
            (("Þetta", "er", "test"), ("f", "s", "n"), ("þetta", "vera", "test")),
            (("Já", "Kannski"), ("a", "a"), ("já", "kannski")),
        ]


def test_dataset_from_file(test_tsv_lemma_file):
    test_ds = FieldedDataset.from_file(test_tsv_lemma_file, fields=(Fields.Tokens, Fields.GoldTags, Fields.GoldLemmas))
    tmp = test_ds[0]
    print(tmp)
    assert tmp == (("Hæ",), ("a",), ("hæ",))
    assert len(test_ds) == 3


def test_dataset_from_file_lemmas(test_tsv_lemma_file):
    test_ds = FieldedDataset.from_file(test_tsv_lemma_file, fields=(Fields.Tokens, Fields.GoldTags, Fields.GoldLemmas))
    tmp = test_ds[0]
    print(tmp)
    assert tmp == (("Hæ",), ("a",), ("hæ",))
    assert len(test_ds) == 3
    assert test_ds.get_field(Fields.GoldLemmas) == (
        ("hæ",),
        ("þetta", "vera", "test"),
        ("já", "kannski"),
    )


def test_add_field(test_tsv_lemma_file):
    test_ds = FieldedDataset.from_file(test_tsv_lemma_file, fields=(Fields.Tokens, Fields.GoldTags, Fields.GoldLemmas))
    test_ds = test_ds.add_field(test_ds.get_field(Fields.GoldTags), Fields.Tags)
    assert len(test_ds.fields) == 4
    for element in test_ds:
        assert len(element) == 4


def test_read_datasets(test_tsv_lemma_file):
    fields = (Fields.Tokens, Fields.GoldTags, Fields.GoldLemmas)
    test_ds = read_datasets([test_tsv_lemma_file], fields=fields)
    assert len(test_ds) == 3


def test_create_mappers_c_w_emb_only(ds):
    w_map = ds.get_vocab_map(special_tokens=VocabMap.UNK_PAD)
    t_map = ds.get_tag_vocab_map(special_tokens=VocabMap.UNK_PAD)
    c_map = ds.get_char_vocab_map(special_tokens=VocabMap.UNK_PAD_EOS_SOS)
    # We always read the tags as well
    assert w_map.w2i["<pad>"] == 0
    assert w_map.w2i["<unk>"] == 1
    assert t_map.w2i["<pad>"] == 0
    assert t_map.w2i["<unk>"] == 1
    assert c_map.w2i["<pad>"] == 0
    assert c_map.w2i["<unk>"] == 1
    assert c_map.w2i["<s>"] == 3
    assert c_map.w2i["</s>"] == 2
    assert w_map.w2i["Hæ"] > 1
    assert "Óla" not in w_map.w2i

    # 4 + pad + unk
    assert len(t_map.w2i) == 6
    # 6 + pad + unk
    assert len(w_map.w2i) == 8
    # 14 + pad + unk + <s> + </s>
    assert len(c_map.w2i) == 18


def test_read_predicted(tagged_test_tsv_file):
    fields = (Fields.Tokens, Fields.GoldTags, Fields.Tags)
    pred_ds = FieldedDataset.from_file(tagged_test_tsv_file, fields=fields)
    assert pred_ds.get_field(Fields.Tokens) == (
        ("Hæ",),
        ("Þetta", "er", "test"),
        ("Já", "Kannski"),
    )
    assert pred_ds.get_field(Fields.GoldTags) == (("a",), ("f", "s", "n"), ("a", "a"))
    assert pred_ds.get_field(Fields.Tags) == (("a",), ("n", "s", "a"), ("a", "a"))


def test_load_dicts(ds):
    _, dicts = load_dicts(ds)
    assert Dicts.Tokens in dicts
    assert Dicts.Chars in dicts
    assert Dicts.FullTag in dicts
    assert len(dicts) == 3


def test_load_dicts_read_datasets(test_tsv_lemma_file):
    train_ds = read_datasets(
        [test_tsv_lemma_file],
    )
    _, dicts = load_dicts(train_ds)


def test_map_to_idx():
    mapping = {"a": 1, "b": 2, "<unk>": 0}
    idxs = map_to_index(("a", "a", "b", "a", "c"), mapping)
    assert idxs.tolist() == [1, 1, 2, 1, 0]


def test_map_to_chars(ds):
    c_map = ds.get_char_vocab_map(special_tokens=VocabMap.UNK_PAD_EOS_SOS)
    idx = -1
    for idx, (sent, _, _) in enumerate(ds):
        if idx == 0:
            mapped = map_to_chars_and_index(sent, c_map.w2i)
            print(mapped)
            assert mapped.shape == (1, 4)  # 1 word, 2 chars + eos + sos
        if idx == 1:
            mapped = map_to_chars_and_index(sent, c_map.w2i)
            print(mapped)
            assert mapped.shape == (3, 7)  # 3 words, 5 chars at most + eos + sos
        if idx == 2:
            mapped = map_to_chars_and_index(sent, c_map.w2i)
            print(mapped)
            assert mapped.shape == (2, 9)  # 2 words, 7 chars at most + eos + sos
    assert idx == 2


def test_map_to_chars_batch(ds):
    sent_batch = [tokens for tokens, _, _ in ds]
    batch = map_to_chars_batch(sent_batch, ds.get_char_vocab_map(VocabMap.UNK_PAD_EOS_SOS).w2i)
    assert batch.shape == (
        3 * 3,
        9,
    )  # 1 + 3 + 2 words, padded and 7 chars at most + eos + sos
    print(batch)


def test_collate_fn(ds):
    _, dicts = load_dicts(ds)
    dl = DataLoader(ds, batch_size=3, collate_fn=ds.collate_fn)  # type: ignore
    assert len(dl) == 1
    for batch in dl:
        assert len(batch) == 5  # The keys
        assert list(batch[BATCH_KEYS.LENGTHS]) == [
            1,
            3,
            2,
        ]  # These sentences are 1, 3, 2 tokens long
        assert list(batch[BATCH_KEYS.TOKEN_CHARS_LENS]) == [
            2,
            5,
            2,
            4,
            2,
            7,
        ]


def test_adjust_lens(ds: FieldedDataset):
    lengths = tuple(1 for _ in range(sum(ds.get_lengths())))
    ds = ds.adjust_lengths(lengths, shorten=True)
    assert ds.get_lengths() == lengths


def test_tokenizer_preprocessing_and_postprocessing(ds: FieldedDataset, electra_model):
    assert len(ds.fields) == len(ds.data)  # sanity check
    assert len(ds) == 3
    assert ds.get_lengths() == (1, 3, 2)
    # be sure that there are too long sentences
    assert any(len(field) > 2 for sentence_fields in ds for field in sentence_fields)
    max_sequence_length = 2 + 2 + 6  # 2 extra for [SEP] and [CLS] and extra defined in function

    wemb = TransformerEmbedding(Modules.BERT, electra_model)
    chunked_ds = chunk_dataset(ds, wemb.tokenizer, max_sequence_length=max_sequence_length)
    assert len(chunked_ds) == 4
    chunked_lengths = chunked_ds.get_lengths()
    assert chunked_lengths == (1, 2, 1, 2)
    # All should be of acceptable length
    assert all(length <= max_sequence_length for length in chunked_lengths)
    dechunked_ds = dechunk_dataset(ds, chunked_ds)
    dechunked_lengths = dechunked_ds.get_lengths()
    assert dechunked_lengths == ds.get_lengths()


def test_more_tokenization(electra_model):
    max_sequence_length = 512
    wemb = TransformerEmbedding(Modules.BERT, electra_model)
    tok = wemb.tokenizer
    # fmt: off
    test = [('Báðar', 'segjast', 'þær', 'hafa', 'verið', 'látnar', 'matast', 'í', 'eldhúsinu', ',', 'eins', 'og', 'hjúum', 'var', 'gjarnan', 'skipað', 'erlendis', ',', 'og', 'ekki', 'líkað', 'það', 'par', 'vel', 'enda', 'vanar', 'meiri', 'virðingu', 'að', 'heiman', '.')]
    # fmt: on
    lengts = get_adjusted_lengths(
        tuple(test),
        tok,
        max_sequence_length=max_sequence_length,
    )
    assert lengts == (len(test[0]),)
    ds = FieldedDataset((tuple(test),), fields=("tokens",))
    chunked_ds = chunk_dataset(ds, tok, max_sequence_length=max_sequence_length)
    # fmt: off
    test = [('Fræðslu-', 'og', 'kynningarfundur', 'kl.', '14', '.')]
    # fmt: on
    lengts = get_adjusted_lengths(
        tuple(test),
        tok,
        max_sequence_length=max_sequence_length,
    )
    ds = FieldedDataset((tuple(test),), fields=("tokens",))
    chunked_ds = chunk_dataset(ds, tok, max_sequence_length=max_sequence_length)
    # fmt: off
    test = [("9,754", "kl.")]
    # fmt: on
    lengts = get_adjusted_lengths(
        tuple(test),
        tok,
        max_sequence_length=max_sequence_length,
    )
    assert lengts == (len(test[0]),)
    ds = FieldedDataset((tuple(test),), fields=("tokens",))
    chunked_ds = chunk_dataset(ds, tok, max_sequence_length=max_sequence_length)
    # fmt: off
    test = [('Kl.', '9', 'handavinna', ',', 'útskurður', ',', 'fótaaðgerð', 'og', 'hárgreiðsla', '.')]
    # fmt: on
    lengts = get_adjusted_lengths(
        tuple(test),
        tok,
        max_sequence_length=max_sequence_length,
    )
    assert lengts == (len(test[0]),)
    ds = FieldedDataset((tuple(test),), fields=("tokens",))
    chunked_ds = chunk_dataset(ds, tok, max_sequence_length=max_sequence_length)

    # fmt: off
    test = [('ýsa', '36', '36', '36', '257', '9,252', 'Und.', 'þorskur', '62', '62', '62', '276', '17,112', 'Ýsa', '169', '34', '129', '5,967', '767,608', 'Þorskur', '204', '160', '174', '460', '79,831', 'Þykkvalúra', '214', '214', '214', '1,070', 'Samtals', '143', '11,129', '1,592,544', 'FISKMARKAÐUR', 'VESTMANNAEYJA', 'Blálanga', '31', '31', '31', '76', '2,356', 'Grálúða', '180', '177', '178', '44', '7,833', 'Gullkarfi', '87', '87', '87', '124', '10,788', 'Hlýri', '76', '76', '76', '899', '68,325', 'Keila', '32', '32', '32', '41', '1,312', 'Keilubróðir', '4', 'Langa', '74', '67', '72', '1,111', '79,799', 'Lúða', '204', '181', '192', '83', '15,917', 'Skarkoli', '21', '21', '21', '5', '105', 'Skata', '97', '17', '68', '25', '1,705', 'Skötuselur', '181', '140', '165', '576', '94,846', 'Steinbítur', '76', '69', '75', '454', '34,203', 'Stórkjafta', '12', '12', '12', '283', '3,396', 'Ufsi', '35', '25', '34', '5,384', '181,863', 'Und.', 'ufsi', '6', '6', '6', '30', '180', 'Ósundurliðað', '3', 'Ýsa', '55', '67', '89', '6,005', 'Þorskur', '186', '76', '141', '236', '33,375', 'Þykkvalúra', '190', '7', '160', '133', '21,244', 'Samtals', '59', '9,600', '563,252', 'FISKMARKAÐUR', 'ÞÓRSHAFNAR', 'Hlýri', '81', '81', '81', '770', '62,371', 'Langa', '59', '59', '59', '43', '2,537', 'Náskata', '15', '15', '15', '103', '1,545', 'Steinbítur', '70', '70', '70', '131', '9,170', 'Ufsi', '22', '22', '22', '777', '17,094', 'Und.', 'ýsa', '29', '29', '29', '192', '5,568', 'Ýsa', '36', '36', '36', '120', '4,320', 'Samtals', '48', '2,136', '102,605', 'FISKMARKAÐURINN', 'Á', 'SKAGASTRÖND', 'Lúða', '459', '151', '298', '21', '6,251', 'Skata', '47', '47', '47', '12', '564', 'Ufsi', '15', '15', '15', '117', '1,755', 'Und.', 'þorskur', '63', '63', '63', '31,500', 'Ýsa', '172', '69', '133', '800', '106,700', 'Þorskur', '229', '90', '129', '6,690', '862,380', 'Samtals', '124', '8,140', '1,009,150', 'FM', 'PATREKSFJARÐAR', 'Lúða', '161', '161', '161', '24', '3,864', 'Skarkoli', '100', '100', '100', '129', '12,900', 'Steinbítur', '68', '68', '68', '75', '5,100', 'Ufsi', '22', '22', '22', '1,042', '22,924', 'Und.', 'þorskur', '64', '64', '64', '1,258', '80,512', 'Ýsa', '122', '41', '86', '370', '31,937', 'Þorskur', '128', '72', '101', '14,733', '1,493,633', 'Samtals', '94', '17,631')]
    # fmt: on
    lengts = get_adjusted_lengths(
        tuple(test),
        tok,
        max_sequence_length=max_sequence_length,
    )
    assert sum(lengts) == len(test[0])
    ds = FieldedDataset((tuple(test),), fields=("tokens",))
    chunked_ds = chunk_dataset(ds, tok, max_sequence_length=max_sequence_length)
    test = [("Síðan", "kom", "Gern", "hab", "'", "ich", "die", "Frau", "'", "n", "geküßt", "úr")]
    lengts = get_adjusted_lengths(
        tuple(test),
        tok,
        max_sequence_length=max_sequence_length,
    )
    assert sum(lengts) == len(test[0])
    ds = FieldedDataset((tuple(test),), fields=("tokens",))
    chunked_ds = chunk_dataset(ds, tok, max_sequence_length=max_sequence_length)
    # fmt: off
    test = [('qt', '/', 'qt-1', '<', '1', '.')]
    # fmt: on
    lengts = get_adjusted_lengths(
        tuple(test),
        tok,
        max_sequence_length=max_sequence_length,
    )
    assert sum(lengts) == len(test[0])
    ds = FieldedDataset((tuple(test),), fields=("tokens",))
    chunked_ds = chunk_dataset(ds, tok, max_sequence_length=max_sequence_length)
    assert chunked_ds is not None
