"""Compare prioritized Cooperative A* and CBS on the project deck graph."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List

from env.cbs_planner import RouteRequest
from env.project_deck_graph import build_project_deck_layout
from env.traffic_planner import SpaceTimeTrafficPlanner, TimedRoute


def benchmark_requests() -> tuple[RouteRequest, ...]:
    """Return a deterministic bottleneck case on the 45-spot deck."""

    return (
        RouteRequest(
            "first",
            "deck_pathway_middle_3",
            "deck_parking_42",
            0.0,
            (
                "deck_pathway_middle_3",
                "deck_parking_42",
            ),
        ),
        RouteRequest(
            "second",
            "deck_pathway_south_4",
            "deck_parking_7",
            0.0,
            (
                "deck_pathway_south_4",
                "deck_parking_7",
            ),
        ),
    )


def route_text(routes: Iterable[TimedRoute]) -> str:
    return "|".join(
        f"{route.entity_id}:{'>'.join(route.nodes)}"
        for route in routes
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="outputs/cbs_path_planning_comparison.csv",
    )
    args = parser.parse_args()

    layout = build_project_deck_layout(45)
    requests = benchmark_requests()
    sequential_planner = SpaceTimeTrafficPlanner(layout.graph)
    sequential_routes: List[TimedRoute] = []
    for request in requests:
        route = sequential_planner.plan(
            request.entity_id,
            request.source,
            request.target,
            request.start_time,
            request.shared_nodes,
        )
        if route is None:
            raise RuntimeError("prioritized Cooperative A* found no path")
        sequential_routes.append(route)

    cbs_planner = SpaceTimeTrafficPlanner(layout.graph)
    cbs_plan = cbs_planner.plan_batch(requests, reserve=False)
    if cbs_plan is None:
        raise RuntimeError("CBS found no path")

    rows: List[Dict[str, object]] = [
        {
            "planner": "prioritized_cooperative_astar",
            "sum_of_costs": sum(
                route.duration
                for route in sequential_routes
            ),
            "makespan": max(
                route.end_time
                for route in sequential_routes
            ),
            "expanded_nodes": "",
            "routes": route_text(sequential_routes),
        },
        {
            "planner": "cbs",
            "sum_of_costs": cbs_plan.sum_of_costs,
            "makespan": cbs_plan.makespan,
            "expanded_nodes": cbs_plan.expanded_nodes,
            "routes": route_text(cbs_plan.routes.values()),
        },
    ]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(
            f"{row['planner']},"
            f"sum_of_costs,{row['sum_of_costs']},"
            f"makespan,{row['makespan']},"
            f"expanded_nodes,{row['expanded_nodes']}"
        )
    print(f"comparison_csv_written: {output}")


if __name__ == "__main__":
    main()
