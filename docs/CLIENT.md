# Python HTTP client

`GabbyClient` lets another Python application call a local or hosted Gabby service using the
versioned stateless API. The client does not store conversation history. Each call supplies its own
task and optional caller-owned `context`, `memory`, and `metadata`.

The async interface is the primary interface:

```python
import asyncio

from gabby import GabbyClient


async def main() -> None:
    async with GabbyClient(
        "https://agents.example.com", "researcher", bearer_token="read-from-your-secret-store"
    ) as client:
        result = await client.arun(
            "Summarize the latest report",
            context={"project": "northstar"},
            memory={"preferred_format": "brief"},
            include_trace=False,
        )
        print(result.output, result.trace_id)

        async for event in client.astream("Find supporting evidence"):
            if event.type == "text_delta":
                print(event.data.get("text", ""), end="", flush=True)


asyncio.run(main())
```

Synchronous applications can use `run()` and `stream()` with the same request fields. A client must
use either the async or sync interface consistently and must be closed with `aclose()` or `close()`;
both context-manager forms handle cleanup. Clients may provide a bearer token or custom auth headers
for host-specific authentication integrations. The default response limit is 4 MiB and can be
configured with `max_response_bytes`.

Remote service URLs must use HTTPS. Plain HTTP is accepted only for loopback development services.
The API client follows the service's `/v1/agents/{agent_name}/run` and `/stream` contracts and does
not automatically retry requests. For a stream that must survive network interruptions, pass a
unique `idempotency_key` to `astream()` or `stream()`. Every yielded `StreamEvent` carries an
`event_id`; after interruption, call the stream method again with the same request and key plus the
last received `event_id` as `last_event_id`. The service resumes the same transient run while its
bounded in-process journal is retained. Resumption requires routing to the same process; it does not
survive process restarts or service-side retention expiry. Without an idempotency key, a disconnected
stream is cancelled as before. The client leaves retry timing and policy to the caller.
HTTP failures are exposed as `GabbyAPIError` with a status code and the service's sanitized error
type. Stream events arrive as `StreamEvent` objects; an `error` event is yielded to the caller as
part of the stream.

See [API compatibility policy](COMPATIBILITY.md) for the pre-1.0 extension contract and
[API serving](ARCHITECTURE.md#serving) for server-side authentication and deployment boundaries.
