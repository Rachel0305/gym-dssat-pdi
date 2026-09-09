"""Small, dependency-free, mixed-variable NSGA-II implementation.

It is intentionally explicit about variable types. Discrete dose and budget
genes are selected from their legal sets and are never rounded from a
continuous value.
"""

from __future__ import annotations

from copy import deepcopy
import math
import random
from typing import Any, Callable


GENE_TYPES = {
    "n_threshold": "continuous",
    "n_dose": "categorical",
    "n_min_interval_days": "integer",
    "n_season_budget": "categorical",
    "water_threshold": "continuous",
    "irrigation_dose": "categorical",
    "irrigation_min_interval_days": "integer",
    "irrigation_season_budget": "categorical",
}


def sample_genome(space: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    genome: dict[str, Any] = {}
    for name, kind in GENE_TYPES.items():
        bounds = space[name]
        if kind == "continuous":
            genome[name] = float(rng.uniform(float(bounds[0]), float(bounds[1])))
        elif kind == "integer":
            genome[name] = int(rng.randint(int(bounds[0]), int(bounds[1])))
        else:
            genome[name] = rng.choice([int(value) for value in bounds])
    return genome


def validate_genome(genome: dict[str, Any], space: dict[str, Any]) -> None:
    for name, kind in GENE_TYPES.items():
        if name not in genome:
            raise ValueError(f"missing gene: {name}")
        bounds = space[name]
        value = genome[name]
        if kind == "continuous":
            if not float(bounds[0]) <= float(value) <= float(bounds[1]):
                raise ValueError(f"{name} outside bounds")
        elif kind == "integer":
            if int(value) != value or not int(bounds[0]) <= int(value) <= int(bounds[1]):
                raise ValueError(f"{name} outside integer bounds")
        elif int(value) not in {int(item) for item in bounds}:
            raise ValueError(f"{name} is not a legal categorical value")


def crossover(a: dict[str, Any], b: dict[str, Any], space: dict[str, Any], rng: random.Random) -> tuple[dict[str, Any], dict[str, Any]]:
    c1, c2 = deepcopy(a), deepcopy(b)
    for name, kind in GENE_TYPES.items():
        if rng.random() > 0.5:
            continue
        if kind == "continuous":
            alpha = rng.random()
            c1[name] = alpha * float(a[name]) + (1.0 - alpha) * float(b[name])
            c2[name] = alpha * float(b[name]) + (1.0 - alpha) * float(a[name])
        elif kind == "integer":
            c1[name], c2[name] = int(a[name]), int(b[name])
        else:
            c1[name], c2[name] = int(a[name]), int(b[name])
    validate_genome(c1, space)
    validate_genome(c2, space)
    return c1, c2


def mutate(genome: dict[str, Any], space: dict[str, Any], rng: random.Random, probability: float | None = None) -> dict[str, Any]:
    out = deepcopy(genome)
    p = probability if probability is not None else 1.0 / len(GENE_TYPES)
    for name, kind in GENE_TYPES.items():
        if rng.random() >= p:
            continue
        bounds = space[name]
        if kind == "continuous":
            left, right = float(bounds[0]), float(bounds[1])
            out[name] = float(rng.uniform(left, right))
        elif kind == "integer":
            left, right = int(bounds[0]), int(bounds[1])
            out[name] = int(rng.randint(left, right))
        else:
            choices = [int(value) for value in bounds if int(value) != int(out[name])]
            out[name] = int(rng.choice(choices)) if choices else int(out[name])
    validate_genome(out, space)
    return out


def _feasible(row: dict[str, Any]) -> bool:
    return bool(row.get("valid", False)) and bool(row.get("feasible", False))


def dominates(a: dict[str, Any], b: dict[str, Any]) -> bool:
    af, bf = _feasible(a), _feasible(b)
    if af and not bf:
        return True
    if bf and not af:
        return False
    if not af and not bf:
        return False
    ay, by = float(a["grain_yield_kg_ha"]), float(b["grain_yield_kg_ha"])
    an, bn = float(a["total_nitrogen_kg_ha"]), float(b["total_nitrogen_kg_ha"])
    ai, bi = float(a["total_irrigation_mm"]), float(b["total_irrigation_mm"])
    no_worse = ay >= by and an <= bn and ai <= bi
    strictly_better = ay > by or an < bn or ai < bi
    return no_worse and strictly_better


def nondominated_sort(rows: list[dict[str, Any]]) -> list[list[int]]:
    domination_sets: list[list[int]] = [[] for _ in rows]
    dominated_count = [0 for _ in rows]
    fronts: list[list[int]] = [[]]
    for i, row_i in enumerate(rows):
        for j, row_j in enumerate(rows):
            if i == j:
                continue
            if dominates(row_i, row_j):
                domination_sets[i].append(j)
            elif dominates(row_j, row_i):
                dominated_count[i] += 1
        if dominated_count[i] == 0:
            fronts[0].append(i)
    cursor = 0
    while cursor < len(fronts) and fronts[cursor]:
        next_front: list[int] = []
        for i in fronts[cursor]:
            for j in domination_sets[i]:
                dominated_count[j] -= 1
                if dominated_count[j] == 0:
                    next_front.append(j)
        cursor += 1
        if next_front:
            fronts.append(next_front)
    return fronts


def crowding_distance(rows: list[dict[str, Any]], front: list[int]) -> dict[int, float]:
    distance = {idx: 0.0 for idx in front}
    if len(front) <= 2:
        for idx in front:
            distance[idx] = math.inf
        return distance
    objectives = [
        ("grain_yield_kg_ha", True),
        ("total_nitrogen_kg_ha", False),
        ("total_irrigation_mm", False),
    ]
    for key, maximize in objectives:
        ordered = sorted(front, key=lambda idx: float(rows[idx][key]), reverse=maximize)
        distance[ordered[0]] = math.inf
        distance[ordered[-1]] = math.inf
        low = float(rows[ordered[-1]][key])
        high = float(rows[ordered[0]][key])
        if abs(high - low) < 1e-12:
            continue
        for pos in range(1, len(ordered) - 1):
            if math.isinf(distance[ordered[pos]]):
                continue
            previous_value = float(rows[ordered[pos - 1]][key])
            next_value = float(rows[ordered[pos + 1]][key])
            distance[ordered[pos]] += abs(next_value - previous_value) / abs(high - low)
    return distance


def _tournament(rows: list[dict[str, Any]], ranks: dict[int, int], crowding: dict[int, float], rng: random.Random) -> dict[str, Any]:
    a, b = rng.randrange(len(rows)), rng.randrange(len(rows))
    if ranks[a] < ranks[b] or (ranks[a] == ranks[b] and crowding.get(a, 0.0) > crowding.get(b, 0.0)):
        return deepcopy(rows[a]["genome"])
    return deepcopy(rows[b]["genome"])


def _environmental_selection(rows: list[dict[str, Any]], size: int) -> list[dict[str, Any]]:
    fronts = nondominated_sort(rows)
    selected: list[dict[str, Any]] = []
    for front in fronts:
        if len(selected) + len(front) <= size:
            selected.extend(rows[idx] for idx in front)
            continue
        crowd = crowding_distance(rows, front)
        for idx in sorted(front, key=lambda item: crowd[item], reverse=True)[: size - len(selected)]:
            selected.append(rows[idx])
        break
    return selected


def run_nsga2(
    space: dict[str, Any],
    evaluate: Callable[[dict[str, Any]], dict[str, Any]],
    population_size: int = 32,
    generations: int = 20,
    seed: int = 1,
    initial_genomes: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    genomes = [deepcopy(item) for item in (initial_genomes or [])]
    while len(genomes) < population_size:
        genomes.append(sample_genome(space, rng))
    genomes = genomes[:population_size]
    cache: dict[tuple[tuple[str, Any], ...], dict[str, Any]] = {}
    history: list[dict[str, Any]] = []
    evaluation_counter = 0

    def evaluate_genome(genome: dict[str, Any], generation: int) -> dict[str, Any]:
        nonlocal evaluation_counter
        key = tuple(sorted(genome.items()))
        if key not in cache:
            validate_genome(genome, space)
            result = dict(evaluate(deepcopy(genome)))
            result["genome"] = deepcopy(genome)
            cache[key] = result
            evaluation_counter += 1
        result = deepcopy(cache[key])
        result["generation"] = int(generation)
        result["evaluation_index"] = int(evaluation_counter if key not in cache else list(cache).index(key) + 1)
        history.append(result)
        return result

    population = [evaluate_genome(genome, 0) for genome in genomes]
    for generation in range(1, generations + 1):
        fronts = nondominated_sort(population)
        ranks = {idx: rank for rank, front in enumerate(fronts) for idx in front}
        crowd = {idx: value for front in fronts for idx, value in crowding_distance(population, front).items()}
        offspring: list[dict[str, Any]] = []
        while len(offspring) < population_size:
            parent_a = _tournament(population, ranks, crowd, rng)
            parent_b = _tournament(population, ranks, crowd, rng)
            child_a, child_b = crossover(parent_a, parent_b, space, rng)
            offspring.append(mutate(child_a, space, rng))
            if len(offspring) < population_size:
                offspring.append(mutate(child_b, space, rng))
        evaluated_offspring = [evaluate_genome(genome, generation) for genome in offspring]
        population = _environmental_selection(population + evaluated_offspring, population_size)
    return history, population


def feasible_pareto(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    valid = [row for row in rows if _feasible(row)]
    fronts = nondominated_sort(valid)
    return [valid[idx] for idx in fronts[0]] if fronts else []
