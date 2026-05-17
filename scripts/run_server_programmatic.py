#!/usr/bin/env python3
import sys, time
sys.path.insert(0,'src')
from shl_agent.api import app
import inspect

print('Starting programmatic server runner')
print('app.title:', getattr(app,'title',None))
print('module file:', inspect.getfile(app.__class__))
print('registered routes:')
for r in app.routes:
    print(r.path, getattr(r,'methods', None))

import uvicorn
if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000, log_level='info')
