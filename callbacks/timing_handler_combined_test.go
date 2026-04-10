package callbacks

import (
	"context"
	"sync"
	"testing"

	"github.com/stretchr/testify/require"
)

// TestCombiningHandler_loggingOnly_twoObservers verifies composition without TimingHandler:
// multiple "logging" observers each receive the same chain lifecycle (left-to-right).
func TestCombiningHandler_loggingOnly_twoObservers(t *testing.T) {
	t.Parallel()
	a, b := &chainHookCounter{}, &chainHookCounter{}
	h := CombiningHandler{Callbacks: []Handler{a, b}}
	ctx := context.Background()
	h.HandleChainStart(ctx, map[string]any{"k": "in"})
	h.HandleChainEnd(ctx, map[string]any{"k": "out"})
	require.Equal(t, 1, a.chainStarts)
	require.Equal(t, 1, b.chainStarts)
	require.Equal(t, 1, a.chainEnds)
	require.Equal(t, 1, b.chainEnds)
}

// TestTimingHandler_metricsOnly_sliceRecorder verifies span events are recorded with a no-op inner;
// no secondary "logging" handler — only metrics (SliceRecorder).
func TestTimingHandler_metricsOnly_sliceRecorder(t *testing.T) {
	t.Parallel()
	rec := &SliceRecorder{}
	th := NewTimingRecorder(t, rec, SimpleHandler{})
	ctx := context.Background()
	th.HandleChainStart(ctx, map[string]any{"x": 1})
	th.HandleChainEnd(ctx, map[string]any{"y": 2})
	var starts, ends int
	for _, e := range rec.Events {
		if e.Name == "chain" && e.Op == SpanOpStart {
			starts++
		}
		if e.Name == "chain" && e.Op == SpanOpEnd {
			ends++
		}
	}
	require.Equal(t, 1, starts)
	require.Equal(t, 1, ends)
}

// TestTimingHandler_combined_metricsAndLoggingObservers stacks TimingHandler (metrics) outermost
// with CombiningHandler of two logging observers as Inner — recorder and both hooks must see the chain.
func TestTimingHandler_combined_metricsAndLoggingObservers(t *testing.T) {
	t.Parallel()
	rec := &SliceRecorder{}
	a, b := &chainHookCounter{}, &chainHookCounter{}
	inner := CombiningHandler{Callbacks: []Handler{a, b}}
	th := NewTimingRecorder(t, rec, inner)
	ctx := context.Background()
	th.HandleChainStart(ctx, map[string]any{"in": true})
	th.HandleChainEnd(ctx, map[string]any{"out": true})

	var chainSpanEvents int
	for _, e := range rec.Events {
		if e.Name == "chain" {
			chainSpanEvents++
		}
	}
	require.Equal(t, 2, chainSpanEvents, "chain span events: %+v", rec.Events)
	require.Equal(t, 1, a.chainStarts)
	require.Equal(t, 1, b.chainStarts)
	require.Equal(t, 1, a.chainEnds)
	require.Equal(t, 1, b.chainEnds)
}

// orderAssertingInner records a failure reason if the inner runs before metrics wrote the expected span
// (same checks as require, but the handler cannot access *testing.T).
type orderAssertingInner struct {
	SimpleHandler
	rec *SliceRecorder
	err string
}

func (o *orderAssertingInner) HandleChainStart(ctx context.Context, inputs map[string]any) {
	if o.rec == nil {
		o.err = "rec nil"
		return
	}
	var saw bool
	for _, e := range o.rec.Events {
		if e.Name == "chain" && e.Op == SpanOpStart {
			saw = true
			break
		}
	}
	if !saw {
		o.err = "inner HandleChainStart ran before metrics recorded chain start"
	}
}

func (o *orderAssertingInner) HandleChainEnd(ctx context.Context, outputs map[string]any) {
	var sawStart, sawEnd bool
	for _, e := range o.rec.Events {
		if e.Name != "chain" {
			continue
		}
		switch e.Op {
		case SpanOpStart:
			sawStart = true
		case SpanOpEnd:
			sawEnd = true
		}
	}
	if !sawStart || !sawEnd {
		o.err = "inner HandleChainEnd ran before metrics recorded chain end"
	}
}

// TestTimingHandler_combined_metricsRecordedBeforeDeferredInner verifies the deferred inner runs
// after the TimingHandler body, so SliceRecorder already holds the corresponding span event.
func TestTimingHandler_combined_metricsRecordedBeforeDeferredInner(t *testing.T) {
	t.Parallel()
	rec := &SliceRecorder{}
	inner := &orderAssertingInner{rec: rec}
	th := NewTimingRecorder(t, rec, inner)
	ctx := context.Background()
	th.HandleChainStart(ctx, nil)
	require.Empty(t, inner.err)
	th.HandleChainEnd(ctx, nil)
	require.Empty(t, inner.err)
}

// TestSliceRecorder_concurrentRecord ensures the test helper is safe under concurrent Record calls.
func TestSliceRecorder_concurrentRecord(t *testing.T) {
	t.Parallel()
	var sr SliceRecorder
	ctx := context.Background()
	var wg sync.WaitGroup
	for i := 0; i < 100; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			sr.Record(ctx, SpanEvent{Name: "x", Op: SpanOpInstant})
		}(i)
	}
	wg.Wait()
	require.Len(t, sr.Events, 100)
}

// chainHookCounter stands in for a logging or tracing handler that only cares about chain hooks.
type chainHookCounter struct {
	SimpleHandler
	chainStarts int
	chainEnds   int
}

func (c *chainHookCounter) HandleChainStart(ctx context.Context, inputs map[string]any) {
	c.chainStarts++
}

func (c *chainHookCounter) HandleChainEnd(ctx context.Context, outputs map[string]any) {
	c.chainEnds++
}
