"""File arguments: str / Path / list / tuple / None are all understood, nothing is split into characters."""
import pathlib

import pytest

from triggerkit.FileIO.paths import normalize_files


def test_normalize_files_accepts_the_usual_forms(tmp_path):
    p = tmp_path / "a.h5"
    assert normalize_files(None) == []
    assert normalize_files("a.h5") == ["a.h5"]                       # NOT ["a", ".", "h", "5"]
    assert normalize_files(p) == [str(p)]
    assert normalize_files([p, "b.h5"]) == [str(p), "b.h5"]
    assert normalize_files(("a.h5", "b.h5")) == ["a.h5", "b.h5"]
    assert normalize_files(x for x in ("a.h5", p)) == ["a.h5", str(p)]
    assert normalize_files([]) == []
    out = normalize_files(["a.h5"])
    assert isinstance(out, list) and all(type(x) is str for x in out)


@pytest.mark.parametrize("bad", [5, 3.2, {"a": 1}, b"raw.h5", ["a.h5", 7], [b"x.h5"], [None]])
def test_normalize_files_rejects_non_paths_naming_the_argument(bad):
    with pytest.raises(TypeError, match="gamma_files"):
        normalize_files(bad, name="gamma_files")


def test_require_nonempty():
    with pytest.raises(ValueError, match="at least one file"):
        normalize_files([], name="gamma_files", require_nonempty=True)
    with pytest.raises(ValueError):
        normalize_files(None, name="gamma_files", require_nonempty=True)
    assert normalize_files("a.h5", require_nonempty=True) == ["a.h5"]


def test_input_list_is_not_aliased():
    original = ["a.h5"]
    out = normalize_files(original)
    out.append("b.h5")
    assert original == ["a.h5"]
