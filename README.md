# LLM Gateway Router

Model-routing module for an LLM gateway. It covers:

- **Token-aware sliding-window rate limiting**: a token budget per API key (50,000 tokens/min by default), stored in **on-disk SQLite**.
- **Automatic failover**: when the primary returns **429** or takes longer than **3000 ms**, the request goes to the backup provider.
- **Standardized, sanitized error payloads**: upstream stack traces, hosts and provider names never reach clients.

It uses the Python 3.11+ standard library only (`asyncio`, `sqlite3`), so there are no third-party dependencies.

## Run

```bash
uv run python -m unittest discover -s tests -t .   # 38 tests
uv run python demo.py                              # walkthrough of every path
GATEWAY_BOOTSTRAP_KEYS="acme=<your-dev-key>" uv run python -m llm_gateway   # HTTP on :8080
```

```bash
curl -H "Authorization: Bearer <your-dev-key>" \
     -d '{"prompt":"hello","max_tokens":64}' localhost:8080/v1/completions
```

## Layout

```
llm_gateway/
  gateway.py                 LLMGateway: the facade that wires the pipeline together
  config.py                  Frozen, validated config dataclasses
  errors.py                  GatewayError hierarchy + ErrorSanitizer (client-safe payloads)
  models.py                  Principal, CompletionRequest (validation), responses
  tokenizer.py               TokenEstimator protocol + heuristic estimator
  clock.py                   Injectable clock (SystemClock / ManualClock)
  ratelimit/policy.py        SlidingWindowPolicy: pure admission + retry-after maths
  ratelimit/limiter.py       SlidingWindowRateLimiter + ExpiredUsageSweeper
  routing/router.py          FallbackRouter: ordered failover with per-attempt deadline
  providers/base.py          ModelProvider ABC (implement this for OpenAI/Anthropic/...)
  providers/errors.py        Upstream error taxonomy (FailureKind, from_status)
  providers/mock.py          Scriptable MockProvider (429 / hang / 500 / crash)
  storage/database.py        SQLiteDatabase: single-writer thread, WAL, BEGIN IMMEDIATE
  storage/usage_repository.py   token_usage log (reserve / commit / release / evict)
  storage/api_key_repository.py api_keys (SHA-256 hashes only)
  storage/audit_repository.py   request_audit (status, provider, attempts, tokens)
  api/http_server.py         Minimal asyncio HTTP transport (optional)
```

## Request pipeline

```
authenticate -> validate -> reserve(estimate) -> route(primary -> secondary)
             -> commit(actual usage) -> 200
on failure   -> release(reservation) -> sanitized error payload
```

## Design notes

### Rate limiter: state and eviction
- It keeps a **sliding-window log**: one row per request in `token_usage(request_id, key_id, created_at, tokens, state)`.
- **Reserve before calling, reconcile after.** A row is inserted as `reserved` with the estimated cost (`prompt tokens + max_tokens`). On success it becomes `committed` with the provider's actual usage. On failure or cancellation it is deleted. Because in-flight reservations count against the budget straight away, concurrent requests can't jointly overshoot it.
- **Atomic check-and-insert.** Evict, sum and insert all run in a single `BEGIN IMMEDIATE` transaction. This holds across coroutines, and across processes sharing the same DB file (both are tested).
- **One boundary rule for admission and eviction.** An entry counts while `created_at > now - window`, and eviction deletes `created_at <= now - window`. Both use the same rule, so they can never disagree.
- **Two eviction paths.** Each `acquire` evicts expired rows for its own key. A background `ExpiredUsageSweeper` purges keys that have gone idle.
- **Self-healing.** If a process crashes mid-request, its `reserved` row ages out of the window on its own.
- **Accurate `Retry-After`.** It is computed from the oldest entries that must expire before the request fits. A request larger than the whole budget gets a non-retryable `413`.

### Async concurrency and timeout races
- Each attempt runs under `asyncio.timeout(3.0)` **in the caller's task**. The primary is cancelled and fully awaited before the secondary starts. That means no orphaned tasks, and two providers are never billing at the same time.
- A result that lands in the same loop tick as the deadline is discarded, so the outcome is decided exactly once (stress-tested).
- **External cancellation** (client disconnect, shutdown) propagates as `CancelledError`. It never triggers a failover.
- **Cancel during reserve.** The `request_id` is generated before reserving, and the SQLite executor is single-threaded FIFO. So a shielded `release(request_id)` always runs after any in-flight INSERT, and no reservation leaks.
- `commit` is shielded too: tokens that were actually consumed are charged even if the client disconnects.

### Failover and error sanitization
- `failover_on` defaults to `{rate_limited, timeout}` and is configurable. Other failures (such as a 400 from the provider) fail fast rather than double-spending.
- Upstream exceptions are internal (`providers/errors.py`). The full detail is logged with the `request_id`. Clients only ever see gateway-authored messages:

```json
{"error": {"code": "upstream_unavailable", "message": "All model providers are currently unavailable. Please retry.", "request_id": "…"}}
```

| Situation | Status | `code` |
|---|---|---|
| Missing or invalid API key | 401 | `unauthenticated` |
| Bad input | 400 | `invalid_request` |
| Over the window budget | 429 + `Retry-After` | `rate_limit_exceeded` |
| Request larger than the whole budget | 413 | `request_too_large` |
| All providers 429 | 503 | `upstream_unavailable` |
| All providers timed out | 504 | `upstream_timeout` |
| Provider 5xx / adapter crash | 502 | `upstream_error` |
| Provider 4xx | 400 | `upstream_rejected_request` |
| Anything unexpected | 500 | `internal_error` |

### Security
- API keys are stored only as SHA-256 hashes. Unknown keys get a `401`, so a caller can't get a fresh quota just by inventing a key.
- Keys come from `GATEWAY_BOOTSTRAP_KEYS` and are never hard-coded. The HTTP layer caps header and body sizes and read time.
- Failed requests are refunded (the reservation is released). Successful ones are charged their actual usage.

## Extending
Add a real provider by subclassing `ModelProvider` and mapping HTTP statuses with `UpstreamError.from_status(...)`. Don't swallow `CancelledError`. To use a real tokenizer, implement `TokenEstimator.count`.
