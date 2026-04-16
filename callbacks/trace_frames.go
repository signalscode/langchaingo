package callbacks

import (
	"context"
	"sync"
)

// traceFrames holds per-logical-trace stack pairing and LLM stream aggregation for TimingHandler.
// Associate one *traceFrames with a context branch using ContextWithTraceFrames (recommended at
// request ingress), EnsureTraceFrames, or TimingHandler.AutoEnsureTraceFrames.
type traceFrames struct {
	mu     sync.Mutex
	stack  []stackFrame
	stream streamAgg
}

var traceFramesKey = struct{ name string }{"callbacks.traceFrames"}

// ContextWithTraceFrames returns a child of ctx that carries a new *traceFrames value.
// Use at HTTP or RPC entry so concurrent requests each get isolated span pairing and
// stream aggregation while sharing one TimingHandler instance.
func ContextWithTraceFrames(ctx context.Context) context.Context {
	return context.WithValue(ctx, traceFramesKey, newTraceFrames())
}

// EnsureTraceFrames returns ctx if it already carries *traceFrames; otherwise it returns
// a child context with a newly allocated *traceFrames (same as ContextWithTraceFrames).
func EnsureTraceFrames(ctx context.Context) context.Context {
	if traceFramesFromContext(ctx) != nil {
		return ctx
	}
	return ContextWithTraceFrames(ctx)
}

func traceFramesFromContext(ctx context.Context) *traceFrames {
	if ctx == nil {
		return nil
	}
	tf, _ := ctx.Value(traceFramesKey).(*traceFrames)
	return tf
}

func newTraceFrames() *traceFrames {
	return &traceFrames{}
}
