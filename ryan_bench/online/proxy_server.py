# SPDX-License-Identifier: Apache-2.0
"""
Disaggregated-prefill proxy.

Fans each request out to the prefill node (max_tokens=1, fire-and-forget so its
KV cache is produced) and then streams the real generation back from the decode
node. Endpoints are configurable via environment variables:

  PREFILL_URL   default http://localhost:8100/v1/completions
  DECODE_URL    default http://localhost:8200/v1/completions
  PROXY_PORT    default 8000

Run: python proxy_server.py   (then send requests to http://localhost:8000)
"""

import asyncio
import os
import sys
import traceback

import aiohttp
from quart import Quart, make_response, request

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)

PREFILL_URL = os.environ.get("PREFILL_URL", "http://localhost:8100/v1/completions")
DECODE_URL = os.environ.get("DECODE_URL", "http://localhost:8200/v1/completions")
PROXY_PORT = int(os.environ.get("PROXY_PORT", "8000"))

app = Quart(__name__)


async def forward_request(url, data):
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        headers = {
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"
        }
        async with session.post(url=url, json=data, headers=headers) as response:
            if response.status == 200:
                async for chunk_bytes in response.content.iter_chunked(1024):
                    yield chunk_bytes


@app.route('/v1/completions', methods=['POST'])
async def handle_request():
    try:
        original_request_data = await request.get_json()

        # Trigger prefill (only need 1 token) so the KV cache is produced.
        prefill_request = original_request_data.copy()
        prefill_request['max_tokens'] = 1
        asyncio.create_task(forward_request(PREFILL_URL, prefill_request).__anext__())

        # Stream the real generation from the decode node.
        generator = forward_request(DECODE_URL, original_request_data)
        response = await make_response(generator)
        response.timeout = None
        return response

    except Exception as e:
        exc_info = sys.exc_info()
        print("Error occurred in disagg prefill proxy server")
        print(e)
        print("".join(traceback.format_exception(*exc_info)))


if __name__ == '__main__':
    app.run(port=PROXY_PORT)
