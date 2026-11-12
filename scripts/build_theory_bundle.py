"""Export canonical theory/models/limits/visualization as one reading copy."""

import argparse
from datetime import date
import os
from pathlib import Path
import re
import subprocess
import sys

from lightpollution.adapters.storage import file_hash,write_json

ROOT=Path(__file__).resolve().parents[1]
DOCUMENTS=['THEORY.md','MODELS.md','LIMITS.md','VISUALIZATION.md']
TOKEN=re.compile(r'(```[^\n]*\n[\s\S]*?\n```|`[^`\n]*`)|(!?\[[^\]\n]*\]\()([^\s)]+)(\))')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'scratch/documentation-export')
    args=parser.parse_args();output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    chunks=['# LightPollution: теория, модели, ограничения и изображения\n'
        '> Дата создания: '+date.today().isoformat()+'\n\n'
        'Я экспортирую канонические документы для последовательного чтения. '
        'Редактируемые оригиналы остаются в корне репозитория.\n']
    records=[];blocks=[]
    for name in DOCUMENTS:
        source=ROOT/name;raw=source.read_text()
        def rewrite(match):
            if match.group(1):return match.group(0)
            target=match.group(3)
            if target.startswith(('https://','http://','#')):return match.group(0)
            path,separator,fragment=target.partition('#');resolved=(source.parent/path).resolve()
            if match.group(2).startswith('!'):
                target=os.path.relpath(resolved,output)
            else:
                target='https://github.com/WarriorOfTea/LightPollution/blob/main/'+resolved.relative_to(ROOT).as_posix()
                target+=separator+fragment if separator else ''
            return match.group(2)+target+match.group(4)
        changed=TOKEN.sub(rewrite,raw)
        original=re.findall(r'```[^\n]*\n[\s\S]*?\n```',raw)
        assert original==re.findall(r'```[^\n]*\n[\s\S]*?\n```',changed)
        blocks.extend(original);chunks.append(changed)
        records.append({'path':name,'sha256':file_hash(source),'fenced_blocks':len(original)})
    text='\n\n'.join(chunks)
    assert blocks==re.findall(r'```[^\n]*\n[\s\S]*?\n```',text)
    source=output/'documentation.md';source.write_text(text)
    write_json(output/'source-check.json',{'source_documents':records,
        'fenced_code_and_formula_blocks':len(blocks),'all_blocks_preserved_byte_for_byte':True,
        'generated_reading_copy_not_a_second_source_of_truth':True})
    subprocess.run([sys.executable,str(ROOT/'scripts/build_report.py'),'--source',str(source),
        '--output',str(output/'documentation.pdf'),'--date',date.today().isoformat()],check=True)


if __name__=='__main__':main()
