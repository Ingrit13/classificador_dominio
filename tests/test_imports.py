"""Smoke test: pacote importável após instalação em modo editável."""
import importlib


def test_import_package():
    pkg = importlib.import_module("classificador_dominio")
    assert hasattr(pkg, "__version__")


def test_import_pipeline_core():
    from classificador_dominio.pipeline_core import MODELO_PADRAO, treinar_com_csv

    assert isinstance(MODELO_PADRAO, str)
    assert callable(treinar_com_csv)


def test_repo_root():
    from classificador_dominio.paths import repo_root

    r = repo_root()
    assert (r / "src" / "classificador_dominio").is_dir()
