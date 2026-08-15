# -*- coding: utf-8 -*-
"""BUG-1: MD 直传时 node_entry 未回写 md_content，导致下游切分报「文件内容不能为空」。"""
from processor.import_processor.nodes.node_entry import NodeEntry


def test_md_file_sets_md_content(tmp_path):
    md = tmp_path / "manual.md"
    content = "# 标题\n\n这是正文内容。\n"
    md.write_text(content, encoding="utf-8")

    state = {"import_file_path": str(md)}
    result = NodeEntry().process(state)

    assert result["md_content"] == content
    assert result["is_md_read_enabled"] is True
    assert result["md_path"] == str(md)


def test_pdf_file_does_not_set_md_content(tmp_path):
    """PDF 路径不应设置 md_content（由 node_pdf_to_md 负责产出）。"""
    pdf = tmp_path / "manual.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")

    state = {"import_file_path": str(pdf)}
    result = NodeEntry().process(state)

    assert "md_content" not in result
    assert result["is_pdf_read_enabled"] is True
