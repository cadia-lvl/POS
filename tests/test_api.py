"""Test api.py functionality."""
import pos
from pos.core import FieldedDataset


def test_tagger(test_tsv_untagged_file, pretrained_tagger: pos.Tagger):
    """Test all methods of the Tagger."""
    # Tag a single sentence
    tags = pretrained_tagger.tag_sent(("Þetta", "er", "setning", "."))
    assert tags == ("fahen", "sfg3en", "nven", "pl")

    # Tag a correctly formatted file.
    dataset = FieldedDataset.from_file(test_tsv_untagged_file)
    tags = pretrained_tagger.tag_bulk(dataset=dataset)
    print(tags)
    assert tags == (("au",), ("fahen", "sfg3en", "nhen"), ("au", "aa"))


def test_long_sents(pretrained_tagger: pos.Tagger):
    """Test a long sentences."""
    # fmt: off
    test_sents = (['Nú', 'eru', 'til', 'umfjöllunar', 'frumvörp', 'sem', 'varða', 'leikreglur', 'á', 'þessu', 'sviði', ',', 'þ.e.', 'hvernig', 'fara', 'skuli', 'með', 'álitamál', 'varðandi', 'lagningu', 'jarðstrengja', ':', 'Er', 'ekki', 'óeðlilegt', 'að', 'opinber', 'og', 'hálfopinber', 'stofnun', ',', 'eða', 'fyrirtæki', ',', 'standi', 'í', 'slíku', ',', 'séu', 'að', 'reyna', 'að', 'knýja', 'á', 'um', 'framkvæmdir', 'eða', 'undirbúning', 'framkvæmda', 'af', 'þessu', 'tagi', 'akkúrat', 'samtímis', 'því', 'að', 'landsskipulagsstefna', 'er', 'í', 'mótun', 'sem', 'á', 'að', 'leysa', 'svæðisskipulag', 'miðhálendisins', 'af', 'hólmi', 'og', 'Alþingi', 'er', 'að', 'fjalla', 'um', 'lagaumgjörðina', 'sem', 'á', 'að', 'notast', 'við', 'í', 'sambandi', 'við', 'ákvarðanatöku', 'um', 'mál', 'af', 'þessu', 'tagi', ',', 't.d.', 'hvar', 'raflínur', 'eru', 'leiddar', 'í', 'jörð', 'og', 'hvar', 'háspennulínur', 'eru', '?'], ['Er', 'ekki', 'ráðlegt', 'að', 'á', 'eftir', 'A', 'komi', 'B.'])
    # fmt: on
    tags = pretrained_tagger.tag_bulk(dataset=test_sents)
    assert len(tags) == len(test_sents)
    for i in range(len(test_sents)):
        assert len(tags[i]) == len(test_sents[i])
