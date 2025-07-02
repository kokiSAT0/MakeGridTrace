import json
from pathlib import Path
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from src import generator  # noqa: E402


def test_generate_puzzle_structure(tmp_path: Path) -> None:
    puzzle = generator.generate_puzzle(4, 4, seed=0)
    # JSON に変換できるか確認
    data = json.dumps(puzzle)
    assert "clues" in puzzle
    assert "solutionEdges" in puzzle
    assert "cluesFull" in puzzle
    assert puzzle["schemaVersion"] == "2.0"
    assert "loopStats" in puzzle
    assert puzzle["loopStats"]["curveRatio"] >= 0.15
    assert "solverStats" in puzzle
    assert puzzle["solverStats"]["steps"] >= 0
    assert puzzle["solverStats"]["maxDepth"] >= 0
    calc = generator._calculate_clues(
        puzzle["solutionEdges"], generator.PuzzleSize(4, 4)
    )
    assert puzzle["cluesFull"] == calc
    # 一時ファイルに保存し読み込んでみる
    file = tmp_path / "puzzle.json"
    file.write_text(data, encoding="utf-8")
    loaded = json.loads(file.read_text(encoding="utf-8"))
    assert loaded["size"] == {"rows": 4, "cols": 4}
    assert loaded["difficulty"] == "normal"


def test_save_puzzle(tmp_path: Path) -> None:
    puzzle = generator.generate_puzzle(4, 4, difficulty="easy", seed=1)
    path = generator.save_puzzle(puzzle, directory=tmp_path)
    assert path.exists()
    assert path.name == "map_gridtrace.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["id"].startswith("sl_4x4_easy_")
    assert "solverStats" in data


def test_validate_puzzle() -> None:
    puzzle = generator.generate_puzzle(4, 4, seed=0)
    # エラーが出ないことを確認
    generator.validate_puzzle(puzzle)


def test_validate_puzzle_fail() -> None:
    puzzle = generator.generate_puzzle(4, 4, seed=3)
    # ループに含まれる辺を一つ壊して検証エラーを期待
    broken = False
    for r, row in enumerate(puzzle["solutionEdges"]["horizontal"]):
        for c, val in enumerate(row):
            if val:
                puzzle["solutionEdges"]["horizontal"][r][c] = False
                broken = True
                break
        if broken:
            break
    if not broken:
        puzzle["solutionEdges"]["vertical"][0][0] = False
    with pytest.raises(ValueError):
        generator.validate_puzzle(puzzle)


def test_zero_adjacent_fail() -> None:
    puzzle = generator.generate_puzzle(3, 3, difficulty="easy", seed=4)
    # 0 を縦に並べて検証エラーを期待
    puzzle["cluesFull"][0][0] = 0
    puzzle["cluesFull"][1][0] = 0
    with pytest.raises(ValueError):
        generator.validate_puzzle(puzzle)


def test_generate_multiple_and_save(tmp_path: Path) -> None:
    puzzles = generator.generate_multiple_puzzles(3, 3, count_each=1, seed=5)
    assert len(puzzles) == 4
    path = generator.save_puzzles(puzzles, directory=tmp_path)
    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert len(data) == 4


def test_puzzle_to_ascii() -> None:
    puzzle = generator.generate_puzzle(2, 2, difficulty="easy", seed=6)
    ascii_art = generator.puzzle_to_ascii(puzzle)
    assert isinstance(ascii_art, str)
    lines = ascii_art.splitlines()
    # 2x2 の場合は行数が 5 になるはず
    assert len(lines) == 5
    # 1 行目には "+" 記号が含まれる
    assert "+" in lines[0]


def test_generate_puzzle_symmetry() -> None:
    puzzle = generator.generate_puzzle(4, 4, symmetry="rotational", seed=7)
    assert puzzle["symmetry"] == "rotational"
