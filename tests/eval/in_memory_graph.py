"""内存版知识图谱，评测脚本用，不依赖 Neo4j"""

from __future__ import annotations

from collections import deque
from typing import Any


class InMemoryGraph:
    """邻接表存储实体关系，支持多跳子图遍历和最短路径检索"""

    def __init__(self) -> None:
        self._entities: dict[str, dict] = {}
        # 无向邻接表：name -> [(rel_type, target)]
        self._adj: dict[str, list[tuple[str, str]]] = {}

    def add_entity(self, name: str, entity_type: str, description: str = "") -> None:
        self._entities[name] = {"type": entity_type, "description": description}
        self._adj.setdefault(name, [])

    def add_relation(self, head: str, rel_type: str, tail: str) -> None:
        self._adj.setdefault(head, []).append((rel_type, tail))
        self._adj.setdefault(tail, []).append((rel_type, head))

    async def get_neighbors(self, entity_name: str, hops: int = 2) -> list[dict]:
        if entity_name not in self._adj:
            return []
        visited: set[str] = {entity_name}
        records: list[dict] = []
        queue: deque[tuple[str, list[str]]] = deque([(entity_name, [])])

        for _ in range(hops):
            if not queue:
                break
            for _ in range(len(queue)):
                node, rels_so_far = queue.popleft()
                for rel_type, target in self._adj.get(node, []):
                    if target == entity_name:
                        continue
                    ent = self._entities.get(target, {})
                    records.append({
                        "source": entity_name,
                        "relations": rels_so_far + [rel_type],
                        "target": target,
                        "target_type": ent.get("type", ""),
                        "target_desc": ent.get("description", ""),
                    })
                    if target not in visited:
                        visited.add(target)
                        queue.append((target, rels_so_far + [rel_type]))
        return records

    async def execute_cypher(self, cypher: str, params: dict | None = None) -> list[dict]:
        # 评测脚本只用到 shortestPath 查询，用 BFS 求最短路径
        params = params or {}
        if "shortestPath" in cypher and "name_a" in params and "name_b" in params:
            path = self._bfs_shortest(params["name_a"], params["name_b"])
            if path:
                return [{"node_names": path["nodes"], "rel_types": path["rels"]}]
        return []

    def _bfs_shortest(self, start: str, end: str) -> dict | None:
        if start == end:
            return {"nodes": [start], "rels": []}
        visited: set[str] = {start}
        queue: deque[tuple[str, list[str], list[str]]] = deque([(start, [start], [])])
        while queue:
            node, path_nodes, path_rels = queue.popleft()
            for rel_type, target in self._adj.get(node, []):
                if target in visited:
                    continue
                new_nodes = path_nodes + [target]
                new_rels = path_rels + [rel_type]
                if target == end:
                    return {"nodes": new_nodes, "rels": new_rels}
                visited.add(target)
                queue.append((target, new_nodes, new_rels))
        return None

    async def init(self) -> None:
        pass

    async def close(self) -> None:
        pass

    async def get_stats(self) -> dict:
        return {"total_entities": len(self._entities), "total_relations": sum(len(v) for v in self._adj.values()) // 2}
