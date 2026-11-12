"""Create a readable Unicode PDF from the project's research report.

Run: uv run python scripts/build_report.py
The default source is the canonical THEORY.md; explicit report sources remain supported.
"""

import argparse
from datetime import date
from html import escape
from pathlib import Path
import re
from tempfile import TemporaryDirectory

import matplotlib
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle,getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image,LongTable,Paragraph,SimpleDocTemplate,Spacer,TableStyle,XPreformatted

from report_math import MathRenderer

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'THEORY.md'
OUTPUT=ROOT/'scratch/documentation-export/theory.pdf'
WIDTH=A4[0]-36*mm


def inline(value,math=None):
    tokens={}
    def protect(match):
        token=f'@@LPTOKEN{len(tokens)}@@'
        raw=match.group(0)
        if raw.startswith('`'):
            tokens[token]='<font name="DejaVuMono">'+escape(raw[1:-1])+'</font>'
        elif math is not None:
            tokens[token]=math.inline(raw[1:-1])
        else:
            raise ValueError('LaTeX requires the mathematical renderer')
        return token
    value=re.sub(r'`[^`\n]+`|(?<!\$)\$(?!\$)[^$\n]+\$(?!\$)',protect,value)
    if '$' in value:
        raise ValueError('Unclosed LaTeX inline expression')
    value=escape(value)
    def link(match):
        label,url=match.groups()
        if not url.startswith(('https://','http://')):
            path,marker,fragment=url.partition('#')
            relative=(SOURCE.parent/path).resolve().relative_to(ROOT).as_posix()
            url='https://github.com/WarriorOfTea/LightPollution/blob/main/'+relative+(marker+fragment if marker else '')
        return f'<link href="{url}" color="#135D83">{label}</link>'
    value=re.sub(r'\[([^\]]+)\]\(([^)]+)\)',link,value)
    value=re.sub(r'\*\*(.+?)\*\*',r'<b>\1</b>',value)
    for token,replacement in tokens.items():
        value=value.replace(token,replacement)
    return value


def main():
    global SOURCE,OUTPUT
    parser=argparse.ArgumentParser(description='Export a repository Markdown report as a readable Unicode PDF')
    parser.add_argument('--source',type=Path,default=SOURCE)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--date',type=date.fromisoformat,default=date.today())
    parser.add_argument('--math-audit',type=Path,help='Save the rendered LaTeX expressions and their dimensions')
    args=parser.parse_args()
    SOURCE,OUTPUT=args.source.resolve(),args.output.resolve()
    report_date=args.date.isoformat()
    fonts=Path(matplotlib.get_data_path())/'fonts/ttf'
    for name,filename in [('DejaVu','DejaVuSans.ttf'),('DejaVuBold','DejaVuSans-Bold.ttf'),('DejaVuMono','DejaVuSansMono.ttf')]:
        pdfmetrics.registerFont(TTFont(name,str(fonts/filename)))
    pdfmetrics.registerFontFamily('DejaVu',normal='DejaVu',bold='DejaVuBold',italic='DejaVu',boldItalic='DejaVuBold')
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle('ReportBody',fontName='DejaVu',fontSize=9.5,leading=14,spaceAfter=7,alignment=TA_LEFT))
    for level,size in [(1,19),(2,14),(3,11)]:
        styles.add(ParagraphStyle(f'ReportH{level}',parent=styles['ReportBody'],fontName='DejaVuBold',fontSize=size,
                                  leading=size*1.3,spaceBefore=12,spaceAfter=9,keepWithNext=True,textColor=colors.HexColor('#16384b')))
    styles.add(ParagraphStyle('ReportCell',parent=styles['ReportBody'],fontSize=7.5,leading=10.5,spaceAfter=0))
    styles.add(ParagraphStyle('ReportCode',parent=styles['ReportBody'],fontName='DejaVuMono',fontSize=7.5,leading=11))
    story=[]
    temporary=TemporaryDirectory(prefix='lightpollution-math-')
    math=MathRenderer(Path(temporary.name),WIDTH)
    def markup(text):
        return inline(text,math)
    lines=SOURCE.read_text().splitlines()
    i=0
    while i<len(lines):
        line=lines[i].strip()
        if not line:
            i+=1;continue
        if line.startswith('```'):
            block=[];i+=1
            while i<len(lines) and not lines[i].startswith('```'):
                block.append(lines[i]);i+=1
            story.append(XPreformatted(escape('\n'.join(block)),styles['ReportCode']))
            i+=1;continue
        if line=='$$':
            block=[];i+=1
            while i<len(lines) and lines[i].strip()!='$$':
                block.append(lines[i]);i+=1
            if i==len(lines):
                raise ValueError('Unclosed LaTeX display block')
            story.append(math.display('\n'.join(block)))
            i+=1;continue
        image=re.fullmatch(r'!\[([^\]]*)\]\(([^)]+)\)',line)
        if image:
            path=(SOURCE.parent/image.group(2)).resolve()
            picture=Image(str(path))
            picture.drawHeight=WIDTH*picture.imageHeight/picture.imageWidth
            picture.drawWidth=WIDTH
            story.extend([Spacer(1,5*mm),picture,Paragraph(markup(image.group(1)),styles['ReportCell']),Spacer(1,4*mm)])
            i+=1;continue
        heading=re.match(r'^(#{1,3})\s+(.*)$',line)
        if heading:
            story.append(Paragraph(markup(heading.group(2)),styles[f'ReportH{len(heading.group(1))}']))
            i+=1;continue
        if line.startswith('|'):
            rows=[]
            while i<len(lines) and lines[i].strip().startswith('|'):
                cells=[s.strip() for s in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r'[:\-\s]+',cell) for cell in cells):
                    rows.append([Paragraph(markup(cell),styles['ReportCell']) for cell in cells])
                i+=1
            columns=len(rows[0])
            widths=[WIDTH/columns]*columns
            if columns==2: widths=[0.34*WIDTH,0.66*WIDTH]
            table=LongTable(rows,colWidths=widths,repeatRows=1,hAlign='LEFT')
            table.setStyle(TableStyle([
                ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e4eef3')),
                ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f5f7f8')]),
                ('GRID',(0,0),(-1,-1),0.35,colors.HexColor('#c8d4da')),
                ('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),6),
                ('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),
            ]))
            story.extend([table,Spacer(1,3*mm)]);continue
        paragraph=[line.lstrip('> ').removeprefix('- ')]
        bullet=line.startswith('- ')
        i+=1
        while i<len(lines) and lines[i].strip() and not re.match(r'^(#{1,3}\s|\||!\[|```|- |\$\$$)',lines[i].strip()):
            paragraph.append(lines[i].strip());i+=1
        story.append(Paragraph(markup(' '.join(paragraph)),styles['ReportBody'],bulletText='•' if bullet else None))

    def footer(canvas,document):
        canvas.setFont('DejaVu',7)
        canvas.setFillColor(colors.HexColor('#5c7480'))
        canvas.drawString(18*mm,12*mm,'LightPollution · исследовательский прототип · '+report_date)
        canvas.drawRightString(A4[0]-18*mm,12*mm,str(document.page))

    OUTPUT.parent.mkdir(parents=True,exist_ok=True)
    document=SimpleDocTemplate(str(OUTPUT),pagesize=A4,leftMargin=18*mm,rightMargin=18*mm,
                               topMargin=16*mm,bottomMargin=20*mm,
                               title=next((line.removeprefix('# ').strip() for line in lines if line.startswith('# ')),
                                          'LightPollution: исследовательский отчёт'),author='LightPollution')
    document.build(story,onFirstPage=footer,onLaterPages=footer)
    if args.math_audit:
        import json
        records=[{key:value for key,value in record.items() if key!='path'} for record in math.records.values()]
        args.math_audit.parent.mkdir(parents=True,exist_ok=True)
        args.math_audit.write_text(json.dumps({'renderer':'Matplotlib MathText, 600 dpi',
            'unique_formulas':len(records),'expressions':records},ensure_ascii=False,indent=2)+'\n')
    temporary.cleanup()
    print(OUTPUT.relative_to(ROOT) if OUTPUT.is_relative_to(ROOT) else OUTPUT)


if __name__=='__main__':
    main()
