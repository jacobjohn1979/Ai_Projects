#!/bin/bash
echo "=== Checking key functions exist before build ==="
python3 - << 'PYEOF'
content = open('coho_extractor.py').read()
checks = [
    ('_to_date',           '_to_date' in content),
    ('_parse_kbprasac',    '_parse_kbprasac' in content),
    ('_parse_woori',       '_parse_woori' in content),
    ('_parse_hattha',      '_parse_hattha' in content),
    ('_parse_canadia',     '_parse_canadia' in content),
    ('_parse_sathapana',   '_parse_sathapana' in content),
    ('_parse_postbank',    '_parse_postbank' in content),
    ('_parse_maybank',     '_parse_maybank' in content),
    ('_parse_philip',      '_parse_philip' in content),
    ('profiles disabled',  'return []' in content and '_load_profiles_for_text' in content),
    ('TRN_CODE first',     content.find('TRN_CODE') < content.find('ABAAKHPP')),
    ('POST DATE first',    content.find('POST DATE') < content.find('ABAAKHPP')),
]
all_ok = True
for name, ok in checks:
    status = 'OK' if ok else 'MISSING'
    if not ok: all_ok = False
    print(f'  {status}  {name}')
print()
print('SAFE TO BUILD:', 'YES' if all_ok else 'NO - fix issues first')
import sys; sys.exit(0 if all_ok else 1)
PYEOF
