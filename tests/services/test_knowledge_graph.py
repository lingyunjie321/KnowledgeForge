"""KnowledgeGraphService 单测：图查询保留来源字段"""

from __future__ import annotations

from unittest.mock import AsyncMock

from services.knowledge_graph import KnowledgeGraphService


async def test_get_neighbors_returns_relationship_sources():
    service = KnowledgeGraphService()
    service.execute_cypher = AsyncMock(return_value=[])

    await service.get_neighbors("张三")

    cypher = service.execute_cypher.await_args.args[0]
    assert "source_docs" in cypher
    assert "r.source" in cypher
