# Context callbacks: metrics + logging example

This program shows the same **metrics** ([`TimingHandler`](https://pkg.go.dev/github.com/tmc/langchaingo/callbacks#TimingHandler) + [`SliceRecorder`](https://pkg.go.dev/github.com/tmc/langchaingo/callbacks#SliceRecorder)) and **logging** ([`LogHandler`](https://pkg.go.dev/github.com/tmc/langchaingo/callbacks#LogHandler)) stack as [callbacks-metrics-log-example](../callbacks-metrics-log-example), but attaches the composed handler to the **request context** with [`ContextWithHandler`](https://pkg.go.dev/github.com/tmc/langchaingo/callbacks#ContextWithHandler) instead of passing it through every constructor option.

Library code merges context-derived handlers with struct fields using [`EffectiveHandler`](https://pkg.go.dev/github.com/tmc/langchaingo/callbacks#EffectiveHandler) at choke points ([`chains.Call`](https://pkg.go.dev/github.com/tmc/langchaingo/chains#Call), agent [`Executor`](https://pkg.go.dev/github.com/tmc/langchaingo/agents#Executor), [`vectorstores.Retriever.GetRelevantDocuments`](https://pkg.go.dev/github.com/tmc/langchaingo/vectorstores#Retriever.GetRelevantDocuments)).

The small `llmCallbacks` wrapper calls `EffectiveHandler` around `GenerateContent` so **direct** `llms.Model` calls behave like real providers that merge context with model-level handlers. Chains and agents do not use `WithCallback` / `WithCallbacksHandler` here—observability comes from context only.

The example allocates **one** [`SliceRecorder`](https://pkg.go.dev/github.com/tmc/langchaingo/callbacks#SliceRecorder), [`TimingHandler`](https://pkg.go.dev/github.com/tmc/langchaingo/callbacks#TimingHandler), and context (via [`ContextWithHandler`](https://pkg.go.dev/github.com/tmc/langchaingo/callbacks#ContextWithHandler)) for all sections. After printing metrics for each section it calls [`SliceRecorder.Reset`](https://pkg.go.dev/github.com/tmc/langchaingo/callbacks#SliceRecorder.Reset) to clear recorded events while reusing the same backing storage and handler stack.

Sections:

1. Direct LLM call (`HandleLLM*`)
2. Chain (`HandleChain*` + `HandleLLM*`)
3. Retriever (`HandleRetriever*`)
4. One-shot MRKL agent with calculator (`HandleAgent*`, `HandleTool*`, `HandleChain*`, `HandleLLM*`)

Direct [`tools.Tool`](https://pkg.go.dev/github.com/tmc/langchaingo/tools#Tool) calls do **not** merge context handlers in the library; use an executor (section 4) or wrap tools in application code.

## Run

From a clone of this repository (the example `go.mod` uses `replace` to build against the current module root):

```bash
cd examples/context-metrics-log-example
go run .
```

If you copy the example elsewhere, remove the `replace github.com/tmc/langchaingo => ../..` line from `go.mod` and run `go mod tidy`.
