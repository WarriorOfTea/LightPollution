"""Assemble current formal theory, preserving code/formula blocks verbatim."""

import argparse
from datetime import date
from pathlib import Path
import re
import subprocess
import sys

from lightpollution.adapters.storage import file_hash, write_json

ROOT=Path(__file__).resolve().parents[1]
DOCUMENTS=['PHYSICS_AND_MATH_GUIDE.md','PHOTOMETRY_AND_SPECTRA.md',
    'OBSERVATION_MODEL.md','NATURAL_TWILIGHT_SNOW.md',
    'TOPOGRAPHY_AND_NEAR_FIELD.md','KSTARS_INTEGRATION.md','EYE_MODEL.md']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('artifacts/kstars-terrain-near-20261111'))
    args=parser.parse_args()
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    chunks=['# LightPollution 0.6: теория с основ, глаз и интеграция\n> Дата создания: '+date.today().isoformat()+'\n\n'
        'Основы оптики → геометрия → перенос → фотометрия → фон/снег → рельеф и ближний свет → KStars → порог зрения.\n'
        'Это объяснение реализованных приближений; формализм не доказывает точность полевого прогноза.\n']
    records=[];original_blocks=[]
    token=re.compile(r'(```[^\n]*\n[\s\S]*?\n```|`[^`\n]*`)|(\[[^\]\n]+\]\()([^\s)]+)(\))')
    for name in DOCUMENTS:
        source=ROOT/'docs'/name
        raw=source.read_text()
        def rewrite(match):
            if match.group(1):return match.group(0)
            target=match.group(3)
            if target.startswith(('https://','http://','#')):return match.group(0)
            relative,separator,fragment=target.partition('#')
            resolved=(source.parent/relative).resolve().relative_to(ROOT).as_posix()
            return match.group(2)+'https://github.com/WarriorOfTea/LightPollution/blob/main/'+resolved+(separator+fragment if separator else '')+match.group(4)
        changed=token.sub(rewrite,raw)
        originals=re.findall(r'```[^\n]*\n[\s\S]*?\n```',raw)
        assert originals==re.findall(r'```[^\n]*\n[\s\S]*?\n```',changed)
        original_blocks.extend(originals)
        chunks.append(changed)
        records.append({'path':str(source.relative_to(ROOT)),'sha256':file_hash(source),'fenced_blocks':len(originals)})
    text='\n\n'.join(chunks)
    assert original_blocks==re.findall(r'```[^\n]*\n[\s\S]*?\n```',text)
    source=output/'theory-from-zero.md';source.write_text(text)
    write_json(output/'theory-source-check.json',{'source_documents':records,
        'fenced_code_and_formula_blocks':len(original_blocks),'all_blocks_preserved_byte_for_byte':True,
        'link_rewriting_excludes_fenced_and_inline_code':True})
    subprocess.run([sys.executable,str(ROOT/'scripts/build_report.py'),'--source',str(source),
        '--output',str(output/'theory-from-zero.pdf'),'--date',date.today().isoformat()],check=True)


if __name__=='__main__':main()
