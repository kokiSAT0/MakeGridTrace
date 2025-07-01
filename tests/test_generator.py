import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from src import generator  # noqa: E402


def test_generate_puzzle_structure(tmp_path: Path) -> None:
    puzzle = generator.generate_puzzle(4, 4)
    # JSON に変換できるか確認
    data = json.dumps(puzzle)
    assert "clues" in puzzle
    assert "solutionEdges" in puzzle
    # 一時ファイルに保存し読み込んでみる
    file = tmp_path / "puzzle.json"
    file.write_text(data, encoding="utf-8")
    loaded = json.loads(file.read_text(encoding="utf-8"))
    assert loaded["size"] == {"rows": 4, "cols": 4}
    assert loaded["difficulty"] == "normal"


def test_save_puzzle(tmp_path: Path) -> None:
    puzzle = generator.generate_puzzle(4, 4, difficulty="easy")
    path = generator.save_puzzle(puzzle, directory=tmp_path)
    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["id"].startswith("sl_4x4_easy_")
