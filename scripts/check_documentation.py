"""Check canonical documentation, local Markdown links and pinned archive links."""

import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote,urlsplit
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[1]
CANONICAL=['README.md','HISTORY.md','THEORY.md','MODELS.md','LIMITS.md','VISUALIZATION.md']
LINK=re.compile(r'!?\[[^\]\n]*\]\(([^\s)]+)\)')


def prose(text):
    text=re.sub(r'```[^\n]*\n[\s\S]*?\n```','',text)
    return re.sub(r'`[^`\n]*`','',text)


def anchors(text):
    result=set();seen=Counter()
    for heading in re.findall(r'^#{1,6}\s+(.+)$',prose(text),re.M):
        slug=re.sub(r'[^\w\-\s]','',heading.lower()).strip().replace(' ','-')
        number=seen[slug];seen[slug]+=1
        result.add(slug+('-'+str(number) if number else ''))
    result.update(re.findall(r'<a\s+(?:id|name)=["\']([^"\']+)',text))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    tracked=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0')
    # Include newly authored documents before git add, exclude ignored workspaces.
    paths={ROOT/p for p in tracked if p.endswith('.md') and (ROOT/p).is_file()}
    paths.update(ROOT/p for p in CANONICAL)
    paths.update((ROOT/'docs').glob('*.md'));paths.add(ROOT/'artifacts/README.md')
    errors=[];local=0;pinned=set();images=0
    for path in sorted(paths):
        text=prose(path.read_text())
        for match in LINK.finditer(text):
            target=unquote(match.group(1));parts=urlsplit(target)
            if parts.scheme or parts.netloc:
                prefix='https://github.com/WarriorOfTea/LightPollution/blob/'
                if target.startswith(prefix):
                    rest=target[len(prefix):];revision,_,relative=rest.partition('/')
                    if re.fullmatch(r'[0-9a-f]{7,40}',revision):
                        item=(revision,relative.partition('#')[0])
                        if item not in pinned:
                            checked=subprocess.run(['git','cat-file','-e',item[0]+':'+item[1]],cwd=ROOT,
                                stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
                            if checked.returncode:errors.append({'file':str(path.relative_to(ROOT)),'target':target,'reason':'missing pinned Git object'})
                            pinned.add(item)
                continue
            if parts.path=='...':continue  # Literal Markdown syntax example in the audit log.
            resolved=(path.parent/parts.path).resolve() if parts.path else path
            local+=1;images+=match.group(0).startswith('!')
            if not resolved.exists():
                errors.append({'file':str(path.relative_to(ROOT)),'target':target,'reason':'missing local target'})
            elif parts.fragment and resolved.suffix=='.md' and parts.fragment not in anchors(resolved.read_text()):
                errors.append({'file':str(path.relative_to(ROOT)),'target':target,'reason':'missing Markdown anchor'})
    history=prose((ROOT/'HISTORY.md').read_text())
    history=LINK.sub('',history)
    if re.search(r'\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}\s+(?:января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)\b',history):
        errors.append({'file':'HISTORY.md','reason':'calendar date in stage narrative'})
    for name in CANONICAL:
        if not (ROOT/name).is_file():errors.append({'file':name,'reason':'missing canonical document'})
    record={'checked_at':datetime.now(ZoneInfo('Europe/Berlin')).isoformat(timespec='seconds'),
        'canonical_documents':CANONICAL,'markdown_files_checked':len(paths),'local_links_checked':local,
        'local_image_links_checked':images,'pinned_archive_Git_objects_checked':len(pinned),
        'HISTORY_calendar_dates_in_narrative':False if not any(e.get('file')=='HISTORY.md' for e in errors) else None,
        'errors':errors,'external_websites_rechecked':False,'physical_simulations_run':False}
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(record,ensure_ascii=False,indent=2))
    if errors:raise SystemExit(1)


if __name__=='__main__':main()
