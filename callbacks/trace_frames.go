package callbacks

import (
	"context"
	"sync"
	"time"
)

const (
	EventKindChain EventKind = iota
	EventKindLLMGen
	EventKindTool
	EventKindRetriever
	EventKindStreaming // pseudo-kind for LLM stream chunk callbacks (not a stack frame)
	countEventKinds    // countEventKinds is the number of EventKind values
)

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

// traceFrames holds per-logical-trace stack pairing and LLM stream aggregation for TimingHandler.
// Associate one *traceFrames with a context branch using ContextWithTraceFrames (recommended at
// request ingress), EnsureTraceFrames, or TimingHandler.AutoEnsureTraceFrames.
type traceFrames struct {
	mu     sync.Mutex
	stack  []stackFrame
	stream streamAgg
}

func newTraceFrames() *traceFrames {
	return &traceFrames{}
}

func traceFramesFromContext(ctx context.Context) *traceFrames {
	if ctx == nil {
		return nil
	}
	tf, _ := ctx.Value(traceFramesKey).(*traceFrames)
	return tf
}

type stackFrame struct {
	kind  EventKind
	start time.Time
	query string // retriever start query
}

type streamAgg struct {
	firstChunkAt *time.Time
	lastChunkAt  time.Time
	chunkCount   int
	bytesTotal   int
}

// EventKind identifies event types. Used for enabling/disabling callbacks and pairing stack frames.
type EventKind byte

func (k EventKind) ok() bool {
	return k < countEventKinds
}
