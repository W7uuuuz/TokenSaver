import re
import subprocess
import sys
import tempfile

html = open(r'D:\DevWorkSpace\HotMirrorStudio\视频降维技术研究\tools\annotate.html',
            encoding='utf-8').read()
scripts = re.findall(r'<script>(.*?)</script>', html, re.S)
print('inline scripts:', len(scripts))
for i, s in enumerate(scripts):
    if not s.strip():
        continue
    with tempfile.NamedTemporaryFile('w', suffix='.js', delete=False,
                                     encoding='utf-8') as f:
        f.write(s)
        path = f.name
    r = subprocess.run(['node', '--check', path], capture_output=True, text=True)
    if r.returncode == 0:
        print(f'script {i}: OK ({len(s)} chars)')
    else:
        print(f'script {i}: FAIL')
        print(r.stderr[:800])
