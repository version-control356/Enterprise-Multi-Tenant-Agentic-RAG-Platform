import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

async def main():
    from app.database import init_pool, list_tenant_documents
    await init_pool()
    analyst_docs = await list_tenant_documents('tenant_alpha', user_role='analyst')
    admin_docs = await list_tenant_documents('tenant_alpha', user_role='admin')
    print('=== Analyst documents ===')
    print(json.dumps(analyst_docs, indent=2))
    print('=== Admin documents ===')
    print(json.dumps(admin_docs, indent=2))

if __name__ == '__main__':
    asyncio.run(main())
