# Callbacks: metrics + logging example

This program shows how to combine:

- **Metrics** — [`callbacks.TimingHandler`](https://pkg.go.dev/github.com/tmc/langchaingo/callbacks#TimingHandler) with a [`callbacks.SliceRecorder`](https://pkg.go.dev/github.com/tmc/langchaingo/callbacks#SliceRecorder) to collect [`SpanEvent`](https://pkg.go.dev/github.com/tmc/langchaingo/callbacks#SpanEvent) records (chain and LLM spans, durations, attributes).
- **Logging** — [`callbacks.LogHandler`](https://pkg.go.dev/github.com/tmc/langchaingo/callbacks#LogHandler) for human-readable lines on stdout.

`TimingHandler` is constructed **outermost** (`NewTimingHandler(rec, logging)`) so span timestamps include work done by the inner logger, matching the [callbacks guide](../../docs/CALLBACKS.md).

The [`fake`](https://pkg.go.dev/github.com/tmc/langchaingo/llms/fake) LLM does not invoke callbacks; this example uses a small `llmCallbacks` wrapper so behavior matches real providers that call `HandleLLMGenerateContentStart` / `End` around `GenerateContent`.

The example includes four sections:

1. direct LLM model call (`HandleLLMGenerateContentStart` / `End`)
2. chain + model call (`HandleChain*` + `HandleLLM*`)
3. direct tool call (`HandleToolStart` / `HandleToolEnd`)
4. one-shot MRKL agent run with calculator tool (`HandleAgent*`, `HandleTool*`, `HandleChain*`, `HandleLLM*`)

## Run

From a clone of this repository (the example `go.mod` uses `replace` to build against the current version of the parent module):

```bash
cd examples/callbacks-metrics-log-example
go run .
```

If you copy the example elsewhere, remove the `replace github.com/tmc/langchaingo => ../..` line from `go.mod` and run `go mod tidy`.
