package callbacks

import (
	"context"
	"encoding/json"
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
//
// JSON encoding is defined by [SpanEvent.MarshalJSON]. Use [json.Marshal], [MarshalSpanEvents],
// or [SpanEventsMetadataJSON].
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

// MarshalJSON implements [json.Marshaler]. Encoded fields:
//   - name, op: strings
//   - start_at, end_at: UTC RFC3339Nano, omitted if zero
//   - duration: [time.Duration.String], omitted if zero
//   - err: [error.Error], omitted if nil
//   - attrs: string map, omitted if empty/nil
//   - orphan: omitted if false
func (e SpanEvent) MarshalJSON() ([]byte, error) {
	w := spanEventJSON{
		Name:   e.Name,
		Op:     string(e.Op),
		Attrs:  e.Attrs,
		Orphan: e.Orphan,
	}
	if !e.StartAt.IsZero() {
		w.StartAt = e.StartAt.UTC().Format(time.RFC3339Nano)
	}
	if !e.EndAt.IsZero() {
		w.EndAt = e.EndAt.UTC().Format(time.RFC3339Nano)
	}
	if e.Duration != 0 {
		w.Duration = e.Duration.String()
	}
	if e.Err != nil {
		w.Err = e.Err.Error()
	}
	return json.Marshal(w)
}

// spanEventJSON is the on-wire JSON representation for [SpanEvent] (snake_case keys).
type spanEventJSON struct {
	Name     string            `json:"name"`
	Op       string            `json:"op"`
	StartAt  string            `json:"start_at,omitempty"`
	EndAt    string            `json:"end_at,omitempty"`
	Duration string            `json:"duration,omitempty"`
	Err      string            `json:"err,omitempty"`
	Attrs    map[string]string `json:"attrs,omitempty"`
	Orphan   bool              `json:"orphan,omitempty"`
}

// MarshalSpanEvents returns JSON for a slice of [SpanEvent] using [SpanEvent.MarshalJSON].
func MarshalSpanEvents(events []SpanEvent) ([]byte, error) {
	return json.Marshal(events)
}

// SpanEventsMetadataJSON returns span_events as JSON for embedding in other metadata blobs (tests, debugging).
func SpanEventsMetadataJSON(events []SpanEvent) (json.RawMessage, error) {
	b, err := json.Marshal(events)
	if err != nil {
		return nil, err
	}
	return json.RawMessage(b), nil
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
