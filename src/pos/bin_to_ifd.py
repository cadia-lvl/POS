import itertools
import logging
import re
from typing import Optional

log = logging.getLogger(__name__)

ORÐFLOKKUR = {"n", "l", "f", "g", "t", "s", "a", "c", "k", "e", "x", "v", "p", "m"}
KYN = {"k", "v", "h"}
TALA = {"e", "f"}
FALL = {"n", "o", "þ", "e"}
GREINIR = {"g", "-"}
SÉRNAFN = {"s", "-"}
BEYGING = {"s", "v", "o"}
STIG = {"f", "m", "e"}
PERSÓNA = {"1", "2", "3"}
HÁTTUR = {"n", "b", "f", "v", "l", "þ"}
MYND = {"g", "m"}
TÍÐ = {"n", "þ"}
FN_FLOKKUR = {"a", "b", "e", "o", "p", "s", "t"}
TO_FLOKKUR = {"f", "a", "p", "o"}
AO_FLOKKUR = {"a", "f", "u"}
ST_FLOKKUR = {"n", "t", "-"}
SK_FLOKKUR = {"s", "t"}
PL_FLOKKUR = {"l", "k", "g", "a"}


def string_product(*args):
    return ("".join(element) for element in itertools.product(*args))


# Öll hugsanleg mörk. Það þarf ekki að vera að öll mörkin skilgreind hér geti komið fram í íslensku.
# Við skilum alltaf mörkum í fullri lengd.
def no_mörk():
    """Nafnorð."""
    return string_product({"n"}, KYN.union({"-"}), TALA, FALL, GREINIR, SÉRNAFN)


def lo_mörk():
    """Lýsingarorð."""
    return string_product({"l"}, KYN, TALA, FALL, BEYGING, STIG)


def fn_mörk():
    """Fornöfn."""
    return itertools.chain(
        string_product({"f"}, FN_FLOKKUR, KYN, TALA, FALL),
        # Persónufornöfn í 1. og 2. persónu eru kynlaus, en 3. persóna hefur kyn.
        string_product({"f"}, {"p"}, PERSÓNA - {"3"}, TALA, FALL),
    )


def gr_mörk():
    """Greinir."""
    return string_product({"g"}, KYN, TALA, FALL)


def to_mörk():
    """Töluorð."""
    return itertools.chain(
        # Ártöl, prósentur og fjöldatölur
        {"ta---", "tp---", "to---"},
        # Frumtölur
        string_product({"t"}, {"f"}, KYN, TALA, FALL),
    )


def so_mörk():
    """Sagnorð."""
    return itertools.chain(
        # Nafnháttur - nútíð er sleppt og ekki til í þáttíð miðmynd
        {"sng---", "sng--þ", "snm---"},
        # Boðháttur - alltaf 2.p og nútíð
        string_product({"sb"}, MYND, {"2"}, TALA, {"n"}),
        # Lýsingarháttur nútíðar
        string_product({"slg---", "slm---"}),
        # Framsögu- og viðtengingarháttur
        string_product({"s"}, {"f", "v"}, MYND, PERSÓNA, TALA, TÍÐ),
        # Lýsingarháttur þátíðar - hann virðist vera til í nefnifalli, þolfalli og þágufalli. Setjum líka inn eignarfall til að vera viss.
        string_product({"s"}, {"þ"}, MYND, KYN, TALA, FALL),
    )


def ao_mörk():
    """Atviksorð."""
    return itertools.chain(
        string_product({"a"}, AO_FLOKKUR - {"u"}, {"m", "e", "-"}),
        # Upphrópun
        {"au-"},
    )


def st_mörk():
    """Samtengingar."""
    return string_product({"c"}, ST_FLOKKUR)


def greinar_mörk():
    return string_product({"p"}, PL_FLOKKUR)


def sk_mörk():
    """Skammstafanir."""
    return string_product({"k"}, SK_FLOKKUR)


def öll_mörk(strip=True):
    mörk = {
        *no_mörk(),
        *lo_mörk(),
        *fn_mörk(),
        *gr_mörk(),
        *to_mörk(),
        *so_mörk(),
        *ao_mörk(),
        *st_mörk(),
        *greinar_mörk(),
        *sk_mörk(),
        # Erlend orð
        "e",
        # Ógreind
        "x",
        # Vefföng
        "v",
        # Tákn
        "m",
        # Erlend sérnöfn
        "n----s",
    }
    if strip:
        return {strip_mark(mark) for mark in mörk}


def strip_mark(mark: str):
    """Fjarlægir '-' í lok marks."""
    mark = mark.rstrip("-")
    if mark.endswith("-"):
        return strip_mark(mark)
    return mark


def kyn(mork: str) -> str:
    if "KVK" in mork:
        return "v"
    if "KK" in mork:
        return "k"
    if "HK" in mork:
        return "h"
    return ""


def tala(mork: str) -> str:
    if "FT" in mork:
        return "f"
    if "ET" in mork:
        return "e"
    return ""


def fall(mork: str) -> str:
    if "OP" in mork:
        # Ópersónuleg beyging sem stýrir falli - þetta er ekki fall sagnarinnar.
        return ""
    if "NF" in mork:
        return "n"
    if "ÞF" in mork:
        return "o"
    if "ÞGF" in mork:
        return "þ"
    if "EF" in mork:
        return "e"
    return ""


def beyging(mork: str) -> str:
    if "SB" in mork:
        return "s"
    if "VB" in mork:
        return "v"
    return ""


frumstig = re.compile(r"F(ST|SB|VB)")
miðstig = re.compile(r"M(ST|SB|VB)")
efstastig = re.compile(r"E(ST|SB|VB)")


def stig(mork: str) -> str:
    if frumstig.search(mork) is not None:
        return "f"
    if miðstig.search(mork) is not None:
        return "m"
    if efstastig.search(mork) is not None:
        return "e"
    return ""


def pers(mork: str) -> str:
    if "1P" in mork:
        return "1"
    if "2P" in mork:
        return "2"
    if "3P" in mork:
        return "3"
    if "BH" in mork:
        # Boðháttur er alltaf í 2.P
        return "2"
    return ""


def háttur(mork: str) -> str:
    if "LHNT" in mork:
        return "l"
    if "LHÞT" in mork:
        return "þ"
    if "BH" in mork:
        return "b"
    if "NH" in mork:
        return "n"
    if "OSKH" in mork:
        # Skilgreint í BÍN en ónotað
        raise ValueError("Óskháttur er ekki studdur")
    if "VH" in mork:
        return "v"
    if "FH" in mork:
        return "f"
    return ""


def mynd(mork: str) -> str:
    if "GM" in mork:
        return "g"
    if "MM" in mork:
        return "m"
    if "LH" in mork:
        # Allur LH í BÍN er í germynd
        return "g"
    return ""


def tíð(mork: str) -> str:
    if "LH" in mork:
        # Lýsingarháttur hefur ekki tíð í marki
        return ""
    if "NT" in mork:
        return "n"
    if "ÞT" in mork:
        return "þ"
    # Boðháttur er alltaf í nútíð
    if "BH" in mork:
        return "n"
    return ""


def greinir(mork: str) -> str:
    if "gr" in mork:
        return "g"
    return ""


def sérnafn(orðmynd: str) -> str:
    if orðmynd.islower():
        return ""
    return "s"


def pfn_kyn(lemma: str) -> str:
    if lemma in {"ég", "þér", "vér", "þú"}:
        # Ekkert kyn
        return ""
    elif lemma == "hann":
        return "k"
    elif lemma == "hún":
        return "v"
    elif lemma == "það":
        return "h"
    else:
        raise ValueError(f"Unknown {lemma=}")


def pfn_persóna(lemma: str) -> str:
    if lemma in {"ég", "vér"}:
        return "1"
    elif lemma in {"þér", "þú"}:
        return "2"
    else:
        # Upplýsingum um 3.P er sleppt
        return ""


def rt_beyging(lemma: str) -> str:
    # Raðtölurnar fyrsti og annar beygjast sterkt (skv. wikipedia)
    if lemma == "annar" or lemma == "fyrsti":
        return "s"
    return "v"


def óákveðiðfn(mörk: str) -> bool:
    # Við nýtum skilgreininguna sem er notuð í BÍN, þó listinn sé ekki tæmandi.
    return "SERST" in mörk


def ábfn(lemma: str) -> bool:
    return lemma in {"sá", "þessi", "hinn"}


def óákveðið_ábfn(lemma: str) -> bool:
    return lemma in {
        "allnokkur",
        "allur",
        "annar",
        "báðir",
        "einhver",
        "einn",
        "enginn",
        "fáeinir",
        "flestallur",
        "hvorugur",
        "mestallur",
        "neinn",
        "nokkur",
        "sérhver",
        "sinnhver",
        "sinnhvor",
        "sitthvað",
        "sitthver",
        "sitthvor",
        "sínhver",
        "sínhvor",
        "sumur",
        "sami",
        "samur",
        "sjálfur",
        "slíkur",
        "ýmis",
        "þónokkur",
        "þvílíkur",
    }


def eignarfn(lemma: str) -> bool:
    return lemma in {"minn", "þinn", "vor", "sinn"}


def spurnarfn(lemma: str) -> bool:
    # "hver", "hvor" og "hvílíkur" get líka verið óákveðin fornöfn.
    return lemma in {"hver", "hvor", "hvaða", "hvílíkur"}


def fn_flokkur(lemma: str, mörk: str) -> str:
    if óákveðiðfn(mörk):
        return "o"
    if ábfn(lemma):
        return "a"
    if óákveðið_ábfn(lemma):
        return "b"
    if eignarfn(lemma):
        return "e"
    if spurnarfn(lemma):
        return "s"
    # Tilvísunarfornöfnin "sem" og "er" eru svo háð samhengi.
    return ""


def greinir_sérnafn(greinir: str, sérnafn: str) -> str:
    if not greinir and sérnafn:
        return "-s"
    else:
        return greinir + sérnafn


def beyging_stig(beyging: str, stig: str) -> str:
    if beyging == "" and stig == "m":
        # Veik beyging fyrir miðstig
        return "vm"
    return beyging + stig


def parse_bin_str(
    orðmynd: str,
    lemma: str,
    kyn_orðflokkur: str,
    mörk: str,
    samtengingar=Optional[str],
    afturbeygð_fn=Optional[str],
) -> Optional[str]:
    if len(orðmynd.split()) == 2 or lemma in {"vettugi"}:
        log.info(f"Skipping {orðmynd} since it is a part of a compound")
        return None
    if kyn_orðflokkur == "afn":
        # Afturbeygt fornafn er "fp" en krefst kyns, tölu og falls frumlags í setningu.
        return afturbeygð_fn
    elif kyn_orðflokkur == "ao":
        if stig(mörk) == "f":
            return "aa"
        return "aa" + stig(mörk)
    elif kyn_orðflokkur == "fn":
        return (
            "f"
            + fn_flokkur(lemma, mörk)
            + kyn(mörk)
            + pers(mörk)
            + tala(mörk)
            + fall(mörk)
        )
    elif kyn_orðflokkur == "fs":
        return "af"
    elif kyn_orðflokkur == "gr":
        return "g" + kyn(mörk) + tala(mörk) + fall(mörk)
    elif kyn_orðflokkur == "hk":
        return (
            "nh"
            + tala(mörk)
            + fall(mörk)
            + greinir_sérnafn(greinir(mörk), sérnafn(orðmynd))
        )
    elif kyn_orðflokkur == "kk":
        return (
            "nk"
            + tala(mörk)
            + fall(mörk)
            + greinir_sérnafn(greinir(mörk), sérnafn(orðmynd))
        )
    elif kyn_orðflokkur == "kvk":
        return (
            "nv"
            + tala(mörk)
            + fall(mörk)
            + greinir_sérnafn(greinir(mörk), sérnafn(orðmynd))
        )
    elif kyn_orðflokkur == "lo":
        return (
            "l"
            + kyn(mörk)
            + tala(mörk)
            + fall(mörk)
            + beyging_stig(beyging(mörk), stig(mörk))
        )
    elif kyn_orðflokkur == "nhm":
        return "cn"
    elif kyn_orðflokkur == "pfn":
        return "fp" + pfn_kyn(lemma) + pfn_persóna(lemma) + tala(mörk) + fall(mörk)
    elif kyn_orðflokkur == "rt":
        # Raðtölur stigbreytast ekki, svo þær eru alltaf í "frumstigi"
        return "l" + kyn(mörk) + tala(mörk) + fall(mörk) + rt_beyging(lemma) + "f"
    elif kyn_orðflokkur == "so":
        if "SAGNB" in mörk:
            # Sagnbót er túlkað sem lýsingarháttur þátíðar í nefnifall, eintölu, hvorugkyni
            return "sþ" + mynd(mörk) + "hen"
        if "GM-NH-ÞT" in mörk:
            return "sng--þ"
        if "GM-BH-ST" in mörk:
            # Stýfður boðháttur.
            return "sbg2en"
        else:
            return (
                "s"
                + háttur(mörk)
                + mynd(mörk)
                + (pers(mörk) + kyn(mörk))
                + (tala(mörk) + tíð(mörk))
                + fall(mörk)
            )
    elif kyn_orðflokkur == "st":
        # MÍM markamengið skilgreinir semtengingu og tilvísunarsamtengingu og við getum því ekki greint á milli.
        return samtengingar
    elif kyn_orðflokkur == "to":
        if mörk == "OBEYGJANLEGT":
            return "to"
        # Öll önnur töluorð í BÍN eru frumtölur
        return "tf" + kyn(mörk) + tala(mörk) + fall(mörk)
    elif kyn_orðflokkur == "uh":
        return "au"
    else:
        raise ValueError(f"Unknown {kyn_orðflokkur=}")
