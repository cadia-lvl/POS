from pos import morphlex


def test_lookup():
    dmii = morphlex.DMII()
    # Single lemma, multiple tags
    results = dmii.lookup_word("segðu")
    assert len(results) == 2
    # No results
    results = dmii.lookup_word(".")
    assert len(results) == 0
    # Multiple lemmas and tags.
    results = dmii.lookup_word("brúa")
    assert results == (
        ("kvkalmEFFT", "brú"),
        ("soalmGM-NH", "brúa"),
        ("soalmGM-FH-NT-1P-ET", "brúa"),
        ("soalmGM-BH-ST", "brúa"),
        ("soalmGM-FH-NT-3P-FT", "brúa"),
    )
