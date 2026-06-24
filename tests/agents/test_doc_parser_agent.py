"""DocParserAgent 单测：分类、分块、纯文本/Markdown/CSV 解析路径"""

from __future__ import annotations

import pytest

from agents.doc_parser_agent import DocParserAgent, DocType


def test_classify_supported_extensions(mock_llm):
    agent = DocParserAgent()
    assert agent._classify("report.pdf") is DocType.PDF
    assert agent._classify("photo.PNG") is DocType.IMAGE
    assert agent._classify("data.Csv") is DocType.TABLE
    assert agent._classify("notes.MD") is DocType.MARKDOWN
    assert agent._classify("readme.txt") is DocType.TEXT


def test_classify_unknown_extension_returns_unknown(mock_llm):
    agent = DocParserAgent()
    assert agent._classify("doc.docx") is DocType.UNKNOWN
    assert agent._classify("noext") is DocType.UNKNOWN


def test_is_supported_file_boundary():
    assert DocParserAgent.is_supported_file("a.txt") is True
    assert DocParserAgent.is_supported_file("a.docx") is False


def test_make_doc_id_deterministic():
    assert DocParserAgent._make_doc_id("/x/y.txt") == DocParserAgent._make_doc_id("/x/y.txt")
    assert DocParserAgent._make_doc_id("/x/y.txt") != DocParserAgent._make_doc_id("/x/z.txt")


async def test_parse_text_file(tmp_path, mock_llm):
    f = tmp_path / "note.txt"
    f.write_text("知识图谱是结构化知识的表示方式。", encoding="utf-8")

    agent = DocParserAgent()
    chunks = await agent.parse(str(f))
    assert len(chunks) == 1
    assert chunks[0].doc_type is DocType.TEXT
    assert "知识图谱" in chunks[0].content
    assert chunks[0].doc_id == DocParserAgent._make_doc_id(str(f))


async def test_parse_markdown_file(tmp_path, mock_llm):
    f = tmp_path / "readme.md"
    f.write_text("# 标题\n\n正文内容", encoding="utf-8")

    agent = DocParserAgent()
    chunks = await agent.parse(str(f))
    assert len(chunks) == 1
    assert chunks[0].doc_type is DocType.MARKDOWN
    assert "标题" in chunks[0].content


async def test_parse_csv_file(tmp_path, mock_llm):
    f = tmp_path / "data.csv"
    f.write_text("name,age\n张三,30\n李四,25\n", encoding="utf-8")

    agent = DocParserAgent()
    chunks = await agent.parse(str(f))
    assert len(chunks) >= 1
    assert chunks[0].doc_type is DocType.TABLE
    all_text = " ".join(c.content for c in chunks)
    assert "张三" in all_text and "李四" in all_text


async def test_parse_unsupported_raises(tmp_path, mock_llm):
    f = tmp_path / "doc.docx"
    f.write_bytes(b"placeholder")

    agent = DocParserAgent()
    with pytest.raises(ValueError, match="不支持"):
        await agent.parse(str(f))


def test_chunk_texts_overlap_and_index(mock_llm):
    agent = DocParserAgent()
    long_text = "字" * (agent.CHUNK_SIZE * 3)
    chunks = agent._chunk_texts([long_text], "doc1", DocType.TEXT, "src")

    assert len(chunks) >= 3
    indexes = [c.chunk_index for c in chunks]
    assert indexes == list(range(len(chunks)))
    assert all(len(c.content) <= agent.CHUNK_SIZE for c in chunks)


def test_chunk_texts_skips_blank(mock_llm):
    agent = DocParserAgent()
    chunks = agent._chunk_texts(["   \n\n  ", "有效内容"], "d", DocType.TEXT, "s")
    assert len(chunks) == 1
    assert chunks[0].content == "有效内容"


async def test_parse_batch_multiple(tmp_path, mock_llm):
    f1 = tmp_path / "a.txt"
    f1.write_text("内容一", encoding="utf-8")
    f2 = tmp_path / "b.txt"
    f2.write_text("内容二", encoding="utf-8")

    agent = DocParserAgent()
    chunks = await agent.parse_batch([str(f1), str(f2)])
    assert len(chunks) == 2
    contents = [c.content for c in chunks]
    assert "内容一" in contents and "内容二" in contents
