# Copyright (c) Twinkle Contributors. All rights reserved.
"""Private ModelScope Hub search tools used by ``ToolExecutor``."""
from __future__ import annotations

import asyncio


class _SearchTools:

    async def _tool_search_datasets(self, query: str, limit: int = 5) -> dict:
        """Search ModelScope for datasets."""
        return await self._search_hub('datasets', query, limit)

    async def _tool_search_models(self, query: str, limit: int = 5) -> dict:
        """Search ModelScope for models."""
        return await self._search_hub('models', query, limit)

    async def _search_hub(self, resource_type: str, query: str, limit: int) -> dict:
        """Unified ModelScope Hub search for models or datasets."""

        def _search():
            if resource_type == 'datasets':
                return self._search_datasets_impl(query, limit)
            else:
                return self._search_models_impl(query, limit)

        try:
            items = await asyncio.get_event_loop().run_in_executor(None, _search)
            return {'query': query, 'results': items}
        except Exception as e:
            return {'error': f'{resource_type.title()} search failed: {e}'}

    @staticmethod
    def _search_datasets_impl(query: str, limit: int) -> list[dict]:
        """Search datasets via ModelScope SDK (new API)."""
        from modelscope.hub.api import HubApi
        api = HubApi()
        result = api.list_datasets('', search=query, page_size=limit)
        datasets = result.get('datasets', [])
        return [{'id': d.get('id', ''), 'name': d.get('display_name', d.get('id', ''))} for d in datasets]

    @staticmethod
    def _search_models_impl(query: str, limit: int) -> list[dict]:
        """Search models via ModelScope HTTP API (SDK doesn't support search)."""
        import requests
        resp = requests.put(
            'https://modelscope.cn/api/v1/models/',
            json={
                'Name': query,
                'PageSize': limit,
                'PageNumber': 1
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        if not data.get('Success'):
            raise RuntimeError(data.get('Message', 'Unknown error'))
        models = data.get('Data', {}).get('Models', [])
        return [{
            'id': f"{m.get('Path', '')}/{m.get('Name', '')}",
            'name': m.get('ChineseName') or m.get('Name', ''),
        } for m in models]
