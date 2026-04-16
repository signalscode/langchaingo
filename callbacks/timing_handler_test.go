package callbacks

import (
	"context"
	"errors"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/schema"
)

func testCtx(tb testing.TB) context.Context {
	tb.Helper()
	return ContextWithTraceFrames(context.Background())
}

func TestAttrsFromContentResponse_usageNormalizationAndJSON(t *testing.T) {
	t.Parallel()
	res := &llms.ContentResponse{
		Choices: []*llms.ContentChoice{{
			GenerationInfo: map[string]any{
				"input_tokens":  int32(10),
				"output_tokens": int64(5),
				"extra":         map[string]any{"a": 1, "b": []int{2, 3}},
			},
		}},
	}
	attrs := attrsFromContentResponse(res)
	require.NotNil(t, attrs)
	require.Equal(t, "10", attrs["prompt_tokens"])
	require.Equal(t, "5", attrs["completion_tokens"])
	require.Equal(t, "15", attrs["total_tokens"])
	require.Equal(t, "10", attrs["geninfo_input_tokens"])
	require.Equal(t, `{"a":1,"b":[2,3]}`, attrs["geninfo_extra"])
}

func TestAttrsFromContentResponse_providerKeyAliases(t *testing.T) {
	t.Parallel()
	res := &llms.ContentResponse{
		Choices: []*llms.ContentChoice{{
			GenerationInfo: map[string]any{
				"PromptTokens":     20,
				"CompletionTokens": 30,
				"TotalTokens":      50,
			},
		}},
	}
	attrs := attrsFromContentResponse(res)
	require.Equal(t, "20", attrs["prompt_tokens"])
	require.Equal(t, "30", attrs["completion_tokens"])
	require.Equal(t, "50", attrs["total_tokens"])
}

func TestStringifyGenInfo_truncatesLongJSON(t *testing.T) {
	t.Parallel()
	long := strings.Repeat("x", 5000)
	s := stringifyGenInfo(map[string]any{"k": long})
	require.LessOrEqual(t, len(s), stringifyGenInfoMaxLen)
	require.True(t, strings.HasSuffix(s, "..."))
}

func TestTimingHandler_nestedChainAndLLM(t *testing.T) {
	t.Parallel()
	rec := &SliceRecorder{}
	th := NewTimingRecorder(t, rec, SimpleHandler{})
	ctx := testCtx(t)
	// Match legacy monotonic clock start (epoch + 1ms steps) for stable duration assertions.
	fixed := time.Unix(1, 0)
	step := time.Millisecond
	th.Now = func() time.Time {
		tm := fixed
		fixed = fixed.Add(step)
		return tm
	}

	th.HandleChainStart(ctx, map[string]any{"k": "v"})
	th.HandleLLMGenerateContentStart(ctx, nil)
	th.HandleLLMGenerateContentEnd(ctx, &llms.ContentResponse{
		Choices: []*llms.ContentChoice{{
			Content: "ok",
			GenerationInfo: map[string]any{
				"TotalTokens": 42,
			},
		}},
	})
	th.HandleChainEnd(ctx, map[string]any{"out": "x"})

	var names []string
	var ops []SpanOp
	for _, e := range rec.Events {
		names = append(names, e.Name)
		ops = append(ops, e.Op)
	}
	wantNames := []string{"chain", "llm_generate", "llm_generate", "chain"}
	wantOps := []SpanOp{SpanOpStart, SpanOpStart, SpanOpEnd, SpanOpEnd}
	require.Len(t, names, len(wantNames), "events: %+v", rec.Events)
	for i := range wantNames {
		require.Equal(t, wantNames[i], names[i], "idx %d", i)
		require.Equal(t, wantOps[i], ops[i], "idx %d", i)
	}
	llmEnd := rec.Events[2]
	require.Greater(t, llmEnd.Duration, time.Duration(0), "expected positive llm duration, got %v", llmEnd.Duration)
}

func TestTimingHandler_LLMStreamThenEnd(t *testing.T) {
	t.Parallel()
	rec := &SliceRecorder{}
	th := NewTimingRecorder(t, rec)

	ctx := testCtx(t)
	th.HandleLLMGenerateContentStart(ctx, nil)
	th.HandleStreamingFunc(ctx, []byte("a"))
	th.HandleStreamingFunc(ctx, []byte("bc"))
	th.HandleLLMGenerateContentEnd(ctx, &llms.ContentResponse{
		Choices: []*llms.ContentChoice{{Content: "abc"}},
	})

	var chunks int
	for _, e := range rec.Events {
		if e.Op == SpanOpStreamChunk {
			chunks++
		}
	}
	require.Equal(t, 2, chunks, "want 2 stream chunks, got %d events %+v", chunks, rec.Events)
	llmEnd := lastEventNamed(t, rec.Events, "llm_generate", SpanOpEnd)
	require.NotNil(t, llmEnd, "missing llm end")
	require.Equal(t, "2", llmEnd.Attrs["stream_chunks"])
	require.Equal(t, "3", llmEnd.Attrs["stream_bytes"])
	require.NotEmpty(t, llmEnd.Attrs["ttft_ns"])
}

func TestTimingHandler_LLMError(t *testing.T) {
	t.Parallel()
	rec := &SliceRecorder{}
	th := NewTimingRecorder(t, rec)
	ctx := testCtx(t)
	th.HandleLLMGenerateContentStart(ctx, nil)
	errTest := errors.New("boom")
	th.HandleLLMError(ctx, errTest)

	e := rec.Events[len(rec.Events)-1]
	require.Equal(t, SpanOpError, e.Op)
	require.Equal(t, errTest, e.Err)
	require.Equal(t, "llm_generate", e.Name)
}

func TestTimingHandler_toolEndOrphan(t *testing.T) {
	t.Parallel()
	rec := &SliceRecorder{}
	th := NewTimingRecorder(t, rec)
	ctx := testCtx(t)
	th.HandleToolEnd(ctx, "out")

	var orphan bool
	for _, e := range rec.Events {
		if e.Name == "tool" && e.Orphan {
			orphan = true
		}
	}
	require.True(t, orphan, "expected orphan tool end: %+v", rec.Events)
}

func TestTimingHandler_retriever(t *testing.T) {
	t.Parallel()
	rec := &SliceRecorder{}
	th := NewTimingRecorder(t, rec)
	ctx := testCtx(t)
	q := "query"
	th.HandleRetrieverStart(ctx, q)
	docs := []schema.Document{{PageContent: "x"}}
	th.HandleRetrieverEnd(ctx, q, docs)

	require.Len(t, rec.Events, 2, "events %+v", rec.Events)
	require.Equal(t, SpanOpEnd, rec.Events[1].Op)
	require.False(t, rec.Events[1].Orphan)
}

func TestTimingHandler_chainErrorOrphan(t *testing.T) {
	t.Parallel()
	rec := &SliceRecorder{}
	th := NewTimingRecorder(t, rec)
	ctx := testCtx(t)
	th.HandleChainError(ctx, errors.New("chainfail"))

	e := rec.Events[0]
	require.True(t, e.Orphan)
	require.Equal(t, SpanOpError, e.Op)
}

func TestTimingHandler_disabledNoRecord(t *testing.T) {
	t.Parallel()
	rec := &SliceRecorder{}
	inner := &callCounter{}
	th := &TimingHandler{Inner: inner, Recorder: rec, Enabled: false, Now: time.Now}
	ctx := context.Background()
	th.HandleChainStart(ctx, nil)
	require.Empty(t, rec.Events, "unexpected record")
	require.Equal(t, 1, inner.chainStarts, "inner not called")
}

func TestTimingHandler_streamOrphanWithoutLLM(t *testing.T) {
	t.Parallel()
	rec := &SliceRecorder{}
	th := NewTimingRecorder(t, rec)
	ctx := testCtx(t)
	th.HandleStreamingFunc(ctx, []byte("x"))
	e := rec.Events[0]
	require.Equal(t, SpanOpStreamChunk, e.Op)
	require.True(t, e.Orphan)
}

func TestTimingHandler_SetEvent(t *testing.T) {
	t.Parallel()
	rec := &SliceRecorder{}
	th := NewTimingRecorder(t, rec)
	ctx := testCtx(t)

	th.SetEvent(EventKindChain, true)
	th.HandleChainStart(ctx, map[string]any{"k": 1})
	th.HandleChainEnd(ctx, map[string]any{"o": 1})
	require.Empty(t, rec.Events, "chain suppressed should not record")

	th.SetEvent(EventKindChain, false)
	th.HandleChainStart(ctx, map[string]any{"k": 2})
	th.HandleChainEnd(ctx, map[string]any{"o": 2})
	require.Len(t, rec.Events, 2)

	th.SetEvent(EventKindStreaming, true)
	th.HandleLLMGenerateContentStart(ctx, nil)
	th.HandleStreamingFunc(ctx, []byte("a"))
	th.HandleLLMGenerateContentEnd(ctx, &llms.ContentResponse{
		Choices: []*llms.ContentChoice{{Content: "a"}},
	})
	streamChunks := 0
	for _, e := range rec.Events {
		if e.Name == "llm_stream" {
			streamChunks++
		}
	}
	require.Equal(t, 0, streamChunks, "streaming suppressed should skip llm_stream chunks")
	require.True(t, th.GetEvent(EventKindStreaming))
}

// NewTimingRecorder builds a TimingHandler with a monotonic clock for tests.
// Optional inners are composed like NewTimingHandler(rec, inners...).
func NewTimingRecorder(t *testing.T, rec *SliceRecorder, inners ...Handler) *TimingHandler {
	t.Helper()
	fixed := time.Unix(1000, 0)
	step := time.Millisecond
	th := NewTimingHandler(rec, inners...)
	th.Now = func() time.Time {
		tm := fixed
		fixed = fixed.Add(step)
		return tm
	}
	return th
}

func lastEventNamed(t *testing.T, events []SpanEvent, name string, op SpanOp) *SpanEvent {
	t.Helper()
	for i := len(events) - 1; i >= 0; i-- {
		if events[i].Name == name && events[i].Op == op {
			return &events[i]
		}
	}
	return nil
}

type callCounter struct {
	SimpleHandler
	chainStarts int
}

func (c *callCounter) HandleChainStart(ctx context.Context, inputs map[string]any) {
	c.chainStarts++
}

func TestTimingHandler_contextMode_concurrentDistinctFramesNoOrphans(t *testing.T) {
	t.Parallel()
	rec := &SliceRecorder{}
	th := NewTimingHandler(rec)
	th.Now = time.Now

	var wg sync.WaitGroup
	const n = 20
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			ctx := ContextWithTraceFrames(context.Background())
			th.HandleChainStart(ctx, map[string]any{"k": 1})
			th.HandleLLMGenerateContentStart(ctx, nil)
			th.HandleLLMGenerateContentEnd(ctx, &llms.ContentResponse{
				Choices: []*llms.ContentChoice{{Content: "x"}},
			})
			th.HandleChainEnd(ctx, map[string]any{"o": 1})
		}()
	}
	wg.Wait()

	var orphans int
	for _, e := range rec.Events {
		if e.Orphan {
			orphans++
		}
	}
	require.Equal(t, 0, orphans, "overlapping traces should not produce orphan spans: %+v", rec.Events)
}

// TestTimingHandler_AutoEnsureTraceFrames_concurrent exercises AutoEnsureTraceFrames under many
// goroutines: each builds a trace using context.Background() on the first callback, then threads
// the auto-installed *traceFrames via the enriched context captured from the inner handler (mirrors
// how callers must propagate ctx after the first stack-aware callback).
func TestTimingHandler_AutoEnsureTraceFrames_concurrent(t *testing.T) {
	t.Parallel()
	const (
		n          = 48
		wantEvents = 5 * n // chain×2 + llm×2 + one stream chunk per goroutine
	)
	rec := &SliceRecorder{}
	var wg sync.WaitGroup
	var nilCtxCount atomic.Int32
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			spy := &contextCaptureInner{}
			th := NewTimingHandler(rec, spy)
			th.AutoEnsureTraceFrames = true
			th.Now = time.Now

			th.HandleChainStart(context.Background(), map[string]any{"id": id})
			if spy.lastCtx == nil {
				nilCtxCount.Add(1)
				return
			}
			ctx := spy.lastCtx

			th.HandleLLMGenerateContentStart(ctx, nil)
			th.HandleStreamingFunc(ctx, []byte("z"))
			th.HandleLLMGenerateContentEnd(ctx, &llms.ContentResponse{
				Choices: []*llms.ContentChoice{{Content: "x"}},
			})
			th.HandleChainEnd(ctx, map[string]any{"out": id})
		}(i)
	}
	wg.Wait()

	require.Equal(t, int32(0), nilCtxCount.Load(), "every chain start should expose enriched ctx to Inner")
	require.Len(t, rec.Events, wantEvents, "event count: %+v", rec.Events)
	var orphans int
	for _, e := range rec.Events {
		if e.Orphan {
			orphans++
		}
	}
	require.Equal(t, 0, orphans, "auto-ensure concurrent traces should not produce orphans: %+v", rec.Events)
}

func TestTimingHandler_contextMode_autoEnsurePropagatesToChildContext(t *testing.T) {
	t.Parallel()
	rec := &SliceRecorder{}
	spy := &contextCaptureInner{}
	th := NewTimingRecorder(t, rec, spy)
	th.AutoEnsureTraceFrames = true

	th.HandleChainStart(context.Background(), nil)
	require.NotNil(t, spy.lastCtx, "inner should see enriched ctx")
	child := context.WithValue(spy.lastCtx, struct{ k string }{"probe"}, "v")
	require.NotNil(t, traceFramesFromContext(child), "child ctx should inherit *traceFrames")

	th.HandleLLMGenerateContentStart(child, nil)
	th.HandleLLMGenerateContentEnd(child, &llms.ContentResponse{
		Choices: []*llms.ContentChoice{{Content: "ok"}},
	})
	th.HandleChainEnd(child, nil)

	var llmEnd *SpanEvent
	for i := len(rec.Events) - 1; i >= 0; i-- {
		if rec.Events[i].Name == "llm_generate" && rec.Events[i].Op == SpanOpEnd {
			llmEnd = &rec.Events[i]
			break
		}
	}
	require.NotNil(t, llmEnd)
	require.False(t, llmEnd.Orphan, "llm end should pair with start on same trace: %+v", rec.Events)
}

func TestTimingHandler_contextMode_wrongRootRecordsOrphanEnd(t *testing.T) {
	t.Parallel()
	rec := &SliceRecorder{}
	th := NewTimingRecorder(t, rec)
	th.AutoEnsureTraceFrames = false

	ctxTrace := ContextWithTraceFrames(context.Background())
	th.HandleChainStart(ctxTrace, nil)
	th.HandleChainEnd(context.Background(), nil)

	var sawOrphanChainEnd bool
	for _, e := range rec.Events {
		if e.Name == "chain" && e.Op == SpanOpEnd && e.Orphan {
			sawOrphanChainEnd = true
		}
	}
	require.True(t, sawOrphanChainEnd, "end on unrelated root should be orphan: %+v", rec.Events)
}

type contextCaptureInner struct {
	SimpleHandler
	lastCtx context.Context
}

func (c *contextCaptureInner) HandleChainStart(ctx context.Context, inputs map[string]any) {
	c.lastCtx = ctx
}
