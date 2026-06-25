"""Neo4j 知识图谱服务：实体/关系 CRUD + 子图检索 + 按来源删除"""

from __future__ import annotations

import logging
import time
from typing import Any

from agents.knowledge_extract_agent import Entity, Relation
from config import settings

logger = logging.getLogger(__name__)

# 关系类型白名单，add_relation 只允许这些类型，避免注入
_ALLOWED_REL_TYPES = {
    "BELONGS_TO", "WORKS_AT", "LOCATED_IN", "DEVELOPED_BY",
    "RELATED_TO", "PART_OF", "USES", "DEPENDS_ON",
}


class KnowledgeGraphService:
    """Neo4j 图谱操作"""

    def __init__(self) -> None:
        self._driver: Any = None

    async def init(self) -> None:
        from neo4j import AsyncGraphDatabase
        self._driver = AsyncGraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
        await self._ensure_indexes()

    async def close(self) -> None:
        if self._driver:
            await self._driver.close()

    async def _ensure_indexes(self) -> None:
        index_queries = [
            "CREATE INDEX IF NOT EXISTS FOR (n:Entity) ON (n.name)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Entity) ON (n.type)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Entity) ON (n.source)",
        ]
        async with self._driver.session() as session:
            for q in index_queries:
                await session.run(q)

    @property
    def is_connected(self) -> bool:
        return self._driver is not None

    async def upsert_entity(self, entity: Entity, version: int = 1, source: str = "") -> None:
        if not self._driver:
            return
        # MERGE 语义：存在则更新版本号和时间戳，不存在则创建
        cypher = """
        MERGE (e:Entity {name: $name})
        ON CREATE SET
            e.type = $type,
            e.description = $description,
            e.version = $version,
            e.source = $source,
            e.created_at = $now,
            e.updated_at = $now
        ON MATCH SET
            e.description = CASE WHEN $description <> '' THEN $description ELSE e.description END,
            e.version = $version,
            e.source = CASE WHEN $source <> '' THEN $source ELSE e.source END,
            e.updated_at = $now
        """
        async with self._driver.session() as session:
            await session.run(cypher, {
                "name": entity.name,
                "type": entity.type,
                "description": entity.description,
                "version": version,
                "source": source,
                "now": int(time.time()),
            })

    async def add_relation(self, relation: Relation, source: str = "") -> None:
        if not self._driver:
            return
        rel_type = relation.relation.upper().replace(" ", "_")
        if rel_type not in _ALLOWED_REL_TYPES:
            logger.warning("关系类型不在白名单，已拒绝: %s", rel_type)
            return
        cypher = f"""
        MATCH (h:Entity {{name: $head}})
        MATCH (t:Entity {{name: $tail}})
        MERGE (h)-[r:{rel_type}]->(t)
        SET r.confidence = $confidence, r.source = $source, r.updated_at = $now
        """
        async with self._driver.session() as session:
            await session.run(cypher, {
                "head": relation.head,
                "tail": relation.tail,
                "confidence": relation.confidence,
                "source": source,
                "now": int(time.time()),
            })

    async def execute_cypher(self, cypher: str, params: dict | None = None) -> list[dict]:
        if not self._driver:
            return []
        async with self._driver.session() as session:
            result = await session.run(cypher, params or {})
            records = await result.data()
            return records

    async def get_entity(self, name: str) -> dict | None:
        cypher = "MATCH (e:Entity {name: $name}) RETURN e"
        records = await self.execute_cypher(cypher, {"name": name})
        return records[0] if records else None

    async def get_neighbors(self, entity_name: str, hops: int = 2) -> list[dict]:
        # 多跳子图遍历，GraphRAG 的核心检索
        cypher = f"""
        MATCH path = (start:Entity {{name: $name}})-[*1..{hops}]-(neighbor)
        RETURN
            start.name AS source,
            [r IN relationships(path) | type(r)] AS relations,
            [r IN relationships(path) | r.source] AS source_docs,
            start.source AS source_doc,
            neighbor.name AS target,
            neighbor.type AS target_type,
            neighbor.description AS target_desc,
            neighbor.source AS target_source
        LIMIT 50
        """
        return await self.execute_cypher(cypher, {"name": entity_name})

    async def search_entities(self, keyword: str, limit: int = 20) -> list[dict]:
        cypher = """
        MATCH (e:Entity)
        WHERE e.name CONTAINS $keyword OR e.description CONTAINS $keyword
        RETURN e.name AS name, e.type AS type, e.description AS description
        LIMIT $limit
        """
        return await self.execute_cypher(cypher, {"keyword": keyword, "limit": limit})

    async def delete_by_source(self, source: str) -> int:
        # 增量更新时先按来源删除旧实体，再重新写入
        cypher = """
        MATCH (e:Entity {source: $source})
        DETACH DELETE e
        RETURN count(e) AS deleted
        """
        records = await self.execute_cypher(cypher, {"source": source})
        return records[0].get("deleted", 0) if records else 0

    async def get_stats(self) -> dict:
        entity_count = await self.execute_cypher("MATCH (e:Entity) RETURN count(e) AS cnt")
        rel_count = await self.execute_cypher("MATCH ()-[r]->() RETURN count(r) AS cnt")
        return {
            "total_entities": entity_count[0]["cnt"] if entity_count else 0,
            "total_relations": rel_count[0]["cnt"] if rel_count else 0,
        }

    async def list_entities(self, limit: int = 200, offset: int = 0) -> list[dict]:
        """分页查实体，给图谱可视化用"""
        cypher = """
        MATCH (e:Entity)
        RETURN e.name AS name, e.type AS type, e.description AS description,
               e.source AS source, e.version AS version
        ORDER BY e.updated_at DESC
        SKIP $skip LIMIT $limit
        """
        return await self.execute_cypher(cypher, {"skip": offset, "limit": limit})

    async def list_relations(self, limit: int = 400, offset: int = 0) -> list[dict]:
        """分页查关系，给图谱可视化用"""
        cypher = """
        MATCH (h:Entity)-[r]->(t:Entity)
        RETURN h.name AS source, type(r) AS relation, t.name AS target,
               r.confidence AS confidence, r.source AS source_doc
        ORDER BY r.updated_at DESC
        SKIP $skip LIMIT $limit
        """
        return await self.execute_cypher(cypher, {"skip": offset, "limit": limit})
