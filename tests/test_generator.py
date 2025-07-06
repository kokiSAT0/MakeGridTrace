import json
import hashlib
from pathlib import Path
import sys
from typing import Any, Dict, cast
import random
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from src import generator  # noqa: E402
from src import puzzle_io  # noqa: E402
from src import validator  # noqa: E402
from src import solver  # noqa: E402
from src import puzzle_builder  # noqa: E402
from src import loop_builder  # noqa: E402
from src import sat_unique  # noqa: E402


def test_generate_puzzle_structure(tmp_path: Path) -> None:
    puzzle = cast(Dict[str, Any], generator.generate_puzzle(4, 4, seed=0))
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
    assert puzzle["solverStats"]["ruleVertex"] >= 0
    assert puzzle["solverStats"]["ruleClue"] >= 0
    assert "qualityScore" in puzzle
    assert 0 <= puzzle["qualityScore"] <= 100
    assert puzzle["generationParams"] == {
        "rows": 4,
        "cols": 4,
        "difficulty": "normal",
        "seed": 0,
        "symmetry": None,
        "theme": None,
        "solverStepLimit": generator.MAX_SOLVER_STEPS,
    }
    assert puzzle["seedHash"] == hashlib.sha256(b"0").hexdigest()
    calc = solver.calculate_clues(puzzle["solutionEdges"], solver.PuzzleSize(4, 4))
    assert puzzle["cluesFull"] == calc
    # 一時ファイルに保存し読み込んでみる
    file = tmp_path / "puzzle.json"
    file.write_text(data, encoding="utf-8")
    loaded = json.loads(file.read_text(encoding="utf-8"))
    assert loaded["size"] == {"rows": 4, "cols": 4}
    assert loaded["difficulty"] == "normal"


def test_save_puzzle(tmp_path: Path) -> None:
    puzzle = cast(
        Dict[str, Any], generator.generate_puzzle(4, 4, difficulty="easy", seed=1)
    )
    path = puzzle_io.save_puzzle(puzzle, directory=tmp_path)
    assert path.exists()
    assert path.name == "map_gridtrace.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["id"].startswith("sl_4x4_easy_")
    assert "solverStats" in data
    assert set(data["solverStats"].keys()) >= {
        "steps",
        "maxDepth",
        "ruleVertex",
        "ruleClue",
    }


def test_generate_puzzle_theme_border() -> None:
    puzzle = cast(
        Dict[str, Any],
        generator.generate_puzzle(3, 3, difficulty="easy", theme="border"),
    )
    validator.validate_puzzle(puzzle)
    assert puzzle["theme"] == "border"
    assert puzzle["generationParams"]["theme"] == "border"
    size = solver.PuzzleSize(3, 3)
    length = sum(sum(row) for row in puzzle["solutionEdges"]["horizontal"]) + sum(
        sum(row) for row in puzzle["solutionEdges"]["vertical"]
    )
    assert length == 2 * (size.rows + size.cols)


def test_generate_puzzle_theme_maze() -> None:
    puzzle = cast(
        Dict[str, Any],
        generator.generate_puzzle(3, 3, difficulty="easy", theme="maze", seed=11),
    )
    validator.validate_puzzle(puzzle)
    assert puzzle["theme"] == "maze"
    assert puzzle["generationParams"]["theme"] == "maze"


def test_generate_puzzle_theme_spiral() -> None:
    puzzle = cast(
        Dict[str, Any],
        generator.generate_puzzle(3, 3, difficulty="easy", theme="spiral", seed=12),
    )
    validator.validate_puzzle(puzzle)
    assert puzzle["theme"] == "spiral"
    assert puzzle["generationParams"]["theme"] == "spiral"


@pytest.mark.slow
def test_generate_puzzle_pattern() -> None:
    """10x10 サイズで pattern テーマが使えるか確認"""
    puzzle = cast(
        Dict[str, Any],
        generator.generate_puzzle(10, 10, difficulty="easy", theme="pattern", seed=21),
    )
    validator.validate_puzzle(puzzle)
    assert puzzle["theme"] == "pattern"


def test_validate_puzzle() -> None:
    puzzle = cast(Dict[str, Any], generator.generate_puzzle(4, 4, seed=0))
    # エラーが出ないことを確認
    validator.validate_puzzle(puzzle)


def test_validate_puzzle_fail() -> None:
    puzzle = cast(Dict[str, Any], generator.generate_puzzle(4, 4, seed=3))
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
        validator.validate_puzzle(puzzle)


def test_zero_adjacent_allowed() -> None:
    size = solver.PuzzleSize(4, 4)
    edges = loop_builder._create_empty_edges(size)
    loop_builder._generate_random_loop(edges, size, random.Random(1))
    clues = solver.calculate_clues(edges, size)
    stats = solver.count_solutions(clues, size, limit=2, return_stats=True)[1]
    puzzle = puzzle_builder._build_puzzle_dict(
        size=size,
        edges=edges,
        clues=[[v for v in row] for row in clues],
        clues_full=clues,
        loop_length=loop_builder._count_edges(edges),
        curve_ratio=loop_builder._calculate_curve_ratio(edges, size),
        difficulty="easy",
        solver_stats=stats,
        symmetry=None,
        theme=None,
        generation_params={},
        seed_hash="x",
        partial=False,
    )
    assert validator.count_zero_adjacent(puzzle["cluesFull"]) > 0
    validator.validate_puzzle(puzzle)


def test_has_zero_adjacent_helper() -> None:
    clues = [[0, 0], [1, 2]]
    assert validator._has_zero_adjacent(clues)
    clues2 = [[0, 1], [2, 3]]
    assert not validator._has_zero_adjacent(clues2)


def test_generator_zero_adjacent_soft() -> None:
    puzzle = cast(Dict[str, Any], generator.generate_puzzle(3, 3, seed=10))
    count = validator.count_zero_adjacent(puzzle["cluesFull"])
    assert count >= 0


def test_validate_puzzle_zero_only_row_fail() -> None:
    size = solver.PuzzleSize(3, 3)
    edges = loop_builder._create_empty_edges(size)
    # 下側中央のセルだけを囲む小さなループを作成
    edges["horizontal"][2][1] = True
    edges["horizontal"][2][2] = True
    edges["horizontal"][3][1] = True
    edges["horizontal"][3][2] = True
    edges["vertical"][2][1] = True
    edges["vertical"][2][2] = True

    clues_full = solver.calculate_clues(edges, size)
    assert clues_full[0] == [0, 0, 0]
    stats = solver.count_solutions(clues_full, size, limit=2, return_stats=True)[1]
    puzzle = puzzle_builder._build_puzzle_dict(
        size=size,
        edges=edges,
        clues=[[v for v in row] for row in clues_full],
        clues_full=clues_full,
        loop_length=loop_builder._count_edges(edges),
        curve_ratio=loop_builder._calculate_curve_ratio(edges, size),
        difficulty="easy",
        solver_stats=stats,
        symmetry=None,
        theme=None,
        generation_params={},
        seed_hash="x",
        partial=False,
    )
    with pytest.raises(ValueError):
        validator.validate_puzzle(puzzle)


def test_generate_puzzle_timeout() -> None:
    with pytest.raises(TimeoutError):
        generator.generate_puzzle(3, 3, timeout_s=0.0)


@pytest.mark.slow
def test_generate_multiple_and_save(tmp_path: Path) -> None:
    puzzles = generator.generate_multiple_puzzles(3, 3, count_each=1, seed=5)
    assert len(puzzles) == 4
    path = puzzle_io.save_puzzles(puzzles, directory=tmp_path)
    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert len(data) == 4


def test_puzzle_to_ascii() -> None:
    puzzle = cast(
        Dict[str, Any], generator.generate_puzzle(2, 2, difficulty="easy", seed=6)
    )
    ascii_art = generator.puzzle_to_ascii(puzzle)
    assert isinstance(ascii_art, str)
    lines = ascii_art.splitlines()
    # 2x2 の場合は行数が 5 になるはず
    assert len(lines) == 5
    # 1 行目には "+" 記号が含まれる
    assert "+" in lines[0]


@pytest.mark.slow
def test_generate_puzzle_symmetry() -> None:
    puzzle = cast(
        Dict[str, Any], generator.generate_puzzle(4, 4, symmetry="rotational", seed=7)
    )
    assert puzzle["symmetry"] == "rotational"
    # 回転対称パズルも検証を行う
    validator.validate_puzzle(puzzle)


@pytest.mark.slow
def test_solution_edges_rotational() -> None:
    puzzle = cast(
        Dict[str, Any], generator.generate_puzzle(4, 4, symmetry="rotational", seed=42)
    )
    # 生成された回転対称パズルを検証
    validator.validate_puzzle(puzzle)
    edges = puzzle["solutionEdges"]
    size = solver.PuzzleSize(4, 4)
    horizontal = edges["horizontal"]
    vertical = edges["vertical"]

    # 水平線が回転対称になっているか確認
    for r in range(size.rows + 1):
        for c in range(size.cols):
            sr = size.rows - r
            sc = size.cols - c - 1
            assert horizontal[r][c] == horizontal[sr][sc]

    # 垂直線も同様にチェック
    for r in range(size.rows):
        for c in range(size.cols + 1):
            sr = size.rows - r - 1
            sc = size.cols - c
            assert vertical[r][c] == vertical[sr][sc]


@pytest.mark.slow
def test_solution_edges_vertical() -> None:
    puzzle = cast(
        Dict[str, Any], generator.generate_puzzle(4, 4, symmetry="vertical", seed=101)
    )
    validator.validate_puzzle(puzzle)
    edges = puzzle["solutionEdges"]
    size = solver.PuzzleSize(4, 4)
    horizontal = edges["horizontal"]
    vertical = edges["vertical"]

    for r in range(size.rows + 1):
        sr = size.rows - r
        for c in range(size.cols):
            assert horizontal[r][c] == horizontal[sr][c]

    for r in range(size.rows):
        sr = size.rows - r - 1
        for c in range(size.cols + 1):
            assert vertical[r][c] == vertical[sr][c]


@pytest.mark.slow
def test_generate_puzzle_parallel() -> None:
    puzzle = cast(
        Dict[str, Any], generator.generate_puzzle_parallel(3, 3, seed=8, jobs=2)
    )
    validator.validate_puzzle(puzzle)


@pytest.mark.slow
def test_generate_multiple_parallel() -> None:
    puzzles = generator.generate_multiple_puzzles(3, 3, count_each=1, seed=8, jobs=2)
    assert len(puzzles) == 4


def test_hint_dispersion_helper() -> None:
    clues = [
        [0, None, 2, None],
        [None, None, None, None],
        [1, None, 3, None],
    ]
    disp = puzzle_builder._calculate_hint_dispersion(clues)
    assert 0.0 < disp < 1.0


@pytest.mark.slow
def test_generation_params_and_seedhash() -> None:
    puzzle = cast(
        Dict[str, Any],
        generator.generate_puzzle(
            3, 3, difficulty="normal", seed=42, symmetry="rotational"
        ),
    )
    expected = {
        "rows": 3,
        "cols": 3,
        "difficulty": "normal",
        "seed": 42,
        "symmetry": "rotational",
        "theme": None,
        "solverStepLimit": generator.MAX_SOLVER_STEPS,
    }
    assert puzzle["generationParams"] == expected
    assert puzzle["seedHash"] == hashlib.sha256(b"42").hexdigest()


@pytest.mark.slow
def test_count_solutions_unique() -> None:
    puzzle = cast(Dict[str, Any], generator.generate_puzzle(3, 3, seed=9))
    size = solver.PuzzleSize(3, 3)
    assert sat_unique.is_unique(puzzle["clues"], size)


def test_reduce_clues_zero_adjacent() -> None:
    size = solver.PuzzleSize(3, 3)
    rng_loop = random.Random(0)
    edges = loop_builder._create_empty_edges(size)
    loop_builder._generate_random_loop(edges, size, rng_loop)
    clues = solver.calculate_clues(edges, size)
    before = validator.count_zero_adjacent(clues)
    reduced = puzzle_builder._reduce_clues(clues, size, random.Random(1), min_hint=1)
    after = validator.count_zero_adjacent(
        [[v if v is not None else -1 for v in row] for row in reduced]
    )
    assert after <= before


def test_optimize_clues_improves_qs() -> None:
    size = solver.PuzzleSize(3, 3)
    rng_loop = random.Random(2)
    edges = loop_builder._create_empty_edges(size)
    loop_builder._generate_random_loop(edges, size, rng_loop)
    clues_full = solver.calculate_clues(edges, size)
    clues = puzzle_builder._reduce_clues(clues_full, size, random.Random(3), min_hint=1)
    sols, stats = solver.count_solutions(clues, size, limit=2, return_stats=True)
    curve_ratio = loop_builder._calculate_curve_ratio(edges, size)
    loop_length = loop_builder._count_edges(edges)
    qs_before = puzzle_builder._calculate_quality_score(
        clues, curve_ratio, stats["steps"], loop_length
    )
    optimized = puzzle_builder._optimize_clues(
        clues,
        clues_full,
        size,
        random.Random(4),
        min_hint=1,
        loop_length=loop_length,
        curve_ratio=curve_ratio,
        solver_steps=stats["steps"],
        iterations=5,
    )
    sols_after = solver.count_solutions(optimized, size, limit=2)
    assert sols_after == 1
    stats_after = solver.count_solutions(optimized, size, limit=2, return_stats=True)[1]
    qs_after = puzzle_builder._calculate_quality_score(
        optimized, curve_ratio, stats_after["steps"], loop_length
    )
    assert qs_after >= qs_before


def test_generate_puzzle_step_limit() -> None:
    puzzle = cast(
        Dict[str, Any],
        generator.generate_puzzle(3, 3, seed=5, solver_step_limit=1000),
    )
    assert puzzle["generationParams"]["solverStepLimit"] == 1000
