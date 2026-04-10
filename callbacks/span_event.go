package callbacks

import (
	"context"
	"sync"
	"time"
)

// SpanOp classifies a span-related observation emitted by TimingHandler.
type SpanOp string

const (
	// SpanOpEnd marks a successful close of a paired lifecycle (pop).
	SpanOpEnd SpanOp = "end"
	// SpanOpError marks an error close of a paired lifecycle (pop).
	SpanOpError SpanOp = "error"
	// SpanOpInstant is for hooks that are not modeled as start/end pairs (e.g. HandleText).
	SpanOpInstant SpanOp = "instant"
	// SpanOpStart marks the opening of a paired lifecycle (push).
	SpanOpStart SpanOp = "start"
	// SpanOpStreamChunk is emitted for each streaming chunk while an LLM span is active.
	SpanOpStreamChunk SpanOp = "stream_chunk"
)

// SpanEvent is a structured observation for logging and metrics backends.
// Name examples: "chain", "llm_generate", "tool", "retriever", "llm_legacy", "stream_orphan".
type SpanEvent struct {
	// Name identifies the logical phase (e.g. chain, llm_generate).
	Name string
	// Op is the operation kind (start, end, error, stream_chunk, instant).
	Op SpanOp
	// StartAt is set for start ops and for end/error (span start time).
	StartAt time.Time
	// EndAt is set for end/error/stream_chunk/instant when relevant.
	EndAt time.Time
	// Duration is set on end/error closes when StartAt is known.
	Duration time.Duration
	// Err is set for error closes or orphan events that represent mismatches.
	Err error
	// Attrs holds lightweight string attributes (model name, tool name, counts, etc.).
	Attrs map[string]string
	// Orphan is true when the event could not be matched to an open span (stack mismatch or stray stream chunk).
	Orphan bool
}

// SpanRecorder receives span events from TimingHandler. Implementations may forward to logs,
// Prometheus, OpenTelemetry, etc. The callbacks package does not require any specific backend.
type SpanRecorder interface {
	Record(ctx context.Context, e SpanEvent)
}

// SliceRecorder appends SpanEvents for tests and debugging. It is safe for concurrent Record calls.
type SliceRecorder struct {
	mu     sync.Mutex
	Events []SpanEvent
}

// Record implements SpanRecorder.
func (s *SliceRecorder) Record(_ context.Context, e SpanEvent) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.Events = append(s.Events, e)
}

// Reset clears recorded events without reallocating the backing array. It is safe to call
// concurrently with [SliceRecorder.Record].
func (s *SliceRecorder) Reset() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.Events = s.Events[:0]
}
