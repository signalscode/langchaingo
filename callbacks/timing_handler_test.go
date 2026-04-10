package callbacks

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/schema"
)

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
	ctx := context.Background()
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

	ctx := context.Background()
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
	ctx := context.Background()
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
	ctx := context.Background()
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
	ctx := context.Background()
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
	ctx := context.Background()
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
	ctx := context.Background()
	th.HandleStreamingFunc(ctx, []byte("x"))
	e := rec.Events[0]
	require.Equal(t, SpanOpStreamChunk, e.Op)
	require.True(t, e.Orphan)
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
