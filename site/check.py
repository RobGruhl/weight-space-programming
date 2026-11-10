#!/usr/bin/env python3
"""Dependency-free checks for the generated public website."""
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse
ROOT=Path(__file__).resolve().parents[1]
DOCS=ROOT/'docs'
class Check(HTMLParser):
    def __init__(self):
        super().__init__(); self.ids=set(); self.anchors=[]
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if 'id' in a:
            assert a['id'] not in self.ids, f'Duplicate id: {a["id"]}'
            self.ids.add(a['id'])
        for key in ('href','src'):
            value=a.get(key,'')
            if not value: continue
            if value.startswith('#'): self.anchors.append(value[1:])
            elif not urlparse(value).scheme:
                assert (DOCS/value).is_file(), f'Missing target: {value}'
        if tag=='script' and 'src' in a:
            assert not urlparse(a['src']).scheme, 'Unexpected external script'
files=list(DOCS.glob('*.html'))
assert len(files)==6, 'Expected six guide pages'
for page in files:
    parser=Check(); parser.feed(page.read_text())
    assert set(parser.anchors)<=parser.ids, f'Unresolved anchor in {page.name}'
    assert '<title>' in page.read_text() and 'name="viewport"' in page.read_text()
print('PASS: six pages, local links/assets, IDs, anchors, viewport and script boundaries')
