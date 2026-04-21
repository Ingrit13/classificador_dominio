"""Caminhos estáveis para artefatos na raiz do repositório (modelo, .env, dados)."""
from pathlib import Path


def repo_root() -> Path:
    """Diretório raiz do projeto (contém ``src/``, ``data/``, ``pyproject.toml``)."""
    return Path(__file__).resolve().parents[2]


def package_dir() -> Path:
    """Diretório do pacote ``classificador_dominio``."""
    return Path(__file__).resolve().parent
