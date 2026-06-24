"""问答 Agent：意图分类 + 查询改写 + 答案生成，检索委托给 GraphRAGPipeline"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from config import settings

logger = logging.getLogger(__name__)


class QueryIntent(str, Enum):
    FACTOID = "factoid"
    ANALYTICAL = "analytical"
    COMPARATIVE = "comparative"
    PROCEDURAL = "procedural"
    EXPLORATORY = "exploratory"


@dataclass
class RetrievedContext:
    content: str
    source: str
    score: float
    retrieval_type: str  # vector | subgraph | path | community
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class QAResult:
    question: str
    answer: str
    contexts: list[RetrievedContext]
    intent: QueryIntent
    confidence: float
    reasoning_steps: list[str] = field(default_factory=list)


INTENT_PROMPT = """\
你是查询意图分类器。根据用户问题返回意图类别（只返回类别名）：
- factoid: 事实型（谁/什么/哪里/何时）
- analytical: 分析型（为什么/怎么理解）
- comparative: 对比型（A和B有什么区别）
- procedural: 流程型（怎么做/步骤）
- exploratory: 探索型（有哪些/概述）
"""

QUERY_REWRITE_PROMPT = """\
你是查询改写专家。将用户问题改写为更适合检索的形式：
1. 提取核心实体和关键词
2. 生成 1-3 个检索查询
3. 返回 JSON: {"queries": ["查询1", "查询2"], "entities": ["实体1"], "keywords": ["关键词1"]}
"""

ANSWER_PROMPT = """\
你是企业知识问答助手。根据检索到的上下文回答用户问题：
1. 答案必须基于提供的上下文，不要编造
2. 上下文不足时明确告知用户
3. 引用信息来源（如 [来源: xxx]）
4. 多信息源综合分析后给出结论
5. 专业、准确、简洁
"""


class QAAgent:
    """意图分类 → 查询改写 → GraphRAG 检索 → 答案生成"""

    def __init__(
        self,
        vector_store: Any = None,
        knowledge_graph: Any = None,
        graph_rag: Any = None,
    ) -> None:
        self.llm = ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            temperature=0,
        )
        self.vector_store = vector_store
        self.knowledge_graph = knowledge_graph
        # 允许外部注入 graph_rag；没传则惰性构造
        self._graph_rag = graph_rag

    @property
    def graph_rag(self):
        if self._graph_rag is None and self.vector_store is not None and self.knowledge_graph is not None:
            from services.graph_rag import GraphRAGPipeline
            self._graph_rag = GraphRAGPipeline(
                vector_store=self.vector_store,
                knowledge_graph=self.knowledge_graph,
            )
        return self._graph_rag

    async def answer(self, question: str) -> QAResult:
        intent = await self._classify_intent(question)
        rewritten = await self._rewrite_query(question)

        contexts: list[RetrievedContext] = []
        if self.graph_rag is not None:
            # 检索全部委托给 GraphRAG 6 步管道
            rag_contexts = await self.graph_rag.retrieve(question)
            contexts = [
                RetrievedContext(
                    content=c.content,
                    source=c.source_type,
                    score=c.score,
                    retrieval_type=c.source_type,
                    metadata=c.metadata,
                )
                for c in rag_contexts
            ]

        top_contexts = contexts[:8]
        answer_text, reasoning = await self._generate_answer(question, top_contexts, intent)

        return QAResult(
            question=question,
            answer=answer_text,
            contexts=top_contexts,
            intent=intent,
            confidence=self._calc_confidence(top_contexts),
            reasoning_steps=reasoning,
        )

    async def _classify_intent(self, question: str) -> QueryIntent:
        messages = [
            SystemMessage(content=INTENT_PROMPT),
            HumanMessage(content=question),
        ]
        resp = await self.llm.ainvoke(messages)
        raw = resp.content.strip().lower()
        for intent in QueryIntent:
            if intent.value in raw:
                return intent
        return QueryIntent.FACTOID

    async def _rewrite_query(self, question: str) -> dict:
        messages = [
            SystemMessage(content=QUERY_REWRITE_PROMPT),
            HumanMessage(content=question),
        ]
        resp = await self.llm.ainvoke(messages)
        try:
            cleaned = resp.content.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0]
            return json.loads(cleaned)
        except (json.JSONDecodeError, IndexError):
            return {"queries": [question], "entities": [], "keywords": []}

    async def _generate_answer(
        self,
        question: str,
        contexts: list[RetrievedContext],
        intent: QueryIntent,
    ) -> tuple[str, list[str]]:
        context_text = "\n\n".join(
            f"[来源 {i+1}: {c.source} | 类型: {c.retrieval_type} | 分数: {c.score:.2f}]\n{c.content}"
            for i, c in enumerate(contexts)
        )
        reasoning_steps = [
            f"识别问题意图: {intent.value}",
            f"GraphRAG 检索到 {len(contexts)} 条上下文",
        ]
        for src in ("vector", "subgraph", "path", "community"):
            n = sum(1 for c in contexts if c.retrieval_type == src)
            if n:
                reasoning_steps.append(f"  - {src}: {n} 条")

        if contexts:
            system_prompt = ANSWER_PROMPT
            user_prompt = f"上下文信息:\n{context_text}\n\n用户问题: {question}"
        else:
            # 没检索到任何上下文，明确告知而非硬编答案
            system_prompt = "你是企业知识问答助手。当前知识库为空或检索无结果，请告知用户暂无相关资料，并基于通用知识谨慎补充。"
            user_prompt = question
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
        resp = await self.llm.ainvoke(messages)
        reasoning_steps.append("答案生成完成")
        return resp.content, reasoning_steps

    @staticmethod
    def _calc_confidence(contexts: list[RetrievedContext]) -> float:
        if not contexts:
            return 0.0
        avg_score = sum(c.score for c in contexts) / len(contexts)
        return min(avg_score, 1.0)
