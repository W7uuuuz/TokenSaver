"""THESIS_DRAFT.md → docx 转换器（python-docx）. 无 Pandoc 时的替代路径.

用法: cd HotMirrorStudio/视频降维技术研究 && PYTHONIOENCODING=utf-8 python tools/md2docx.py
输出: 论文交付.docx（与 THESIS_DRAFT.md 同目录）
"""
from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "THESIS_DRAFT.md"
OUT = ROOT / "论文交付.docx"

CN_FONT = "宋体"
EN_FONT = "Times New Roman"


def set_run_font(run, size=12, bold=False, mono=False):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.name = EN_FONT
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    rfonts.set(qn("w:ascii"), "Consolas" if mono else EN_FONT)
    rfonts.set(qn("w:hAnsi"), "Consolas" if mono else EN_FONT)
    rfonts.set(qn("w:eastAsia"), CN_FONT)


def add_inline(par, text, base_size=12):
    """解析 **bold** 与 `code` 内联格式."""
    tokens = re.split(r"(\*\*.*?\*\*|`[^`]*`)", text)
    for tok in tokens:
        if not tok:
            continue
        if tok.startswith("**") and tok.endswith("**"):
            set_run_font(par.add_run(tok[2:-2]), base_size, bold=True)
        elif tok.startswith("`") and tok.endswith("`"):
            set_run_font(par.add_run(tok[1:-1]), base_size, mono=True)
        else:
            set_run_font(par.add_run(tok), base_size)


def add_table(doc, rows):
    """rows: list[list[str]]，第一行为表头."""
    table = doc.add_table(rows=len(rows), cols=max(len(r) for r in rows))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, row in enumerate(rows):
        for j, raw in enumerate(row):
            cell = table.cell(i, j)
            cell.text = ""
            par = cell.paragraphs[0]
            add_inline(par, raw.strip())
            if i == 0:
                par.alignment = WD_ALIGN_PARAGRAPH.CENTER
            if par.runs:
                set_run_font(par.runs[0], 10.5, bold=(i == 0))


def parse_rows(lines, i):
    """从行 i 起连续 '|' 行解析成 rows；返回 (rows, 下一行索引)."""
    rows = []
    while i < len(lines) and lines[i].strip().startswith("|"):
        cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
        rows.append(cells)
        i += 1
    return rows, i


def main() -> None:
    lines = SRC.read_text(encoding="utf-8").splitlines()
    doc = Document()

    # Normal 样式基础字号
    normal = doc.styles["Normal"]
    normal.font.name = EN_FONT
    normal.font.size = Pt(12)
    normal.element.rPr.rFonts.set(qn("w:eastAsia"), CN_FONT)

    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        if stripped.startswith("```"):
            i += 1
            code_lines = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            i += 1  # 跳过结束 ```
            for cl in code_lines:
                par = doc.add_paragraph()
                set_run_font(par.add_run(cl or " "), 9, mono=True)
            continue

        if stripped.startswith("|"):
            rows, i = parse_rows(lines, i)
            if len(rows) >= 2 and all(re.fullmatch(r":?-+:?", c.replace(" ", "")) for c in rows[1]):
                header, rows = rows[0], rows[2:]
            else:
                header, rows = rows[0], rows[1:]
            add_table(doc, [header] + rows)
            continue

        if stripped.startswith("#"):
            level = len(stripped) - len(stripped.lstrip("#"))
            text = stripped.lstrip("#").strip()
            style = {1: "Title", 2: "Heading 1", 3: "Heading 2", 4: "Heading 3"}.get(
                level, "Heading 3")
            par = doc.add_heading("", level=min(level, 3))
            # 中文字体
            for run in par.runs:
                run.font.name = EN_FONT
                run._element.rPr.rFonts.set(qn("w:eastAsia"), CN_FONT)
            add_inline(par, text)
            if style == "Title":
                par.alignment = WD_ALIGN_PARAGRAPH.CENTER
            i += 1
            continue

        if stripped.startswith(">"):
            text = stripped.lstrip(">").strip()
            par = doc.add_paragraph()
            set_run_font(par.add_run(text), 11)
            par.paragraph_format.left_indent = Pt(18)
            i += 1
            continue

        if stripped.startswith("- "):
            text = stripped[2:]
            par = doc.add_paragraph(style="List Bullet")
            add_inline(par, text)
            i += 1
            continue

        if re.match(r"^\d+\.\s", stripped):
            text = re.sub(r"^\d+\.\s", "", stripped)
            par = doc.add_paragraph(style="List Number")
            add_inline(par, text)
            i += 1
            continue

        # 普通段落
        par = doc.add_paragraph()
        par.paragraph_format.first_line_indent = Pt(24)
        par.paragraph_format.line_spacing = 1.5
        add_inline(par, stripped)
        i += 1

    doc.save(OUT)
    print(f"saved: {OUT}")


if __name__ == "__main__":
    main()
