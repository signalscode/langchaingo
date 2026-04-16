package callbacks

import (
	"context"
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
	"time"

	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/schema"
)

// spanKind identifies stack frames for paired callbacks.
type spanKind int

const (
	spanChain spanKind = iota
	spanLLMGen
	spanTool
	spanRetriever
)

type stackFrame struct {
	kind  spanKind
	start time.Time
	query string // retriever start query
}

type streamAgg struct {
	firstChunkAt *time.Time
	lastChunkAt  time.Time
	chunkCount   int
	bytesTotal   int
}

// TimingHandler wraps an inner Handler and records span timing and streaming metrics via SpanRecorder.
// It implements the full Handler interface and should be used as the outermost handler when you want
// timings to include work performed by inner handlers.
//
// Stack pairing and LLM stream aggregation use *traceFrames stored on context.Context (see
// ContextWithTraceFrames, EnsureTraceFrames). Each logical trace should use a context branch that carries
// those frames so one TimingHandler instance can serve concurrent overlapping traces safely.
type TimingHandler struct {
	Inner    Handler
	Recorder SpanRecorder
	// Enabled gates recording and stack updates. When false, callbacks are forwarded to Inner only.
	Enabled bool
	// Now returns the current time; if nil, time.Now is used. Useful for tests.
	Now func() time.Time

	// AutoEnsureTraceFrames allocates *traceFrames and attaches it with context.WithValue when ctx
	// does not already carry one. Inner receives the enriched ctx. Repeated callbacks must still pass a
	// context that inherits that value; calling with context.Background() each time cannot pair spans.
	AutoEnsureTraceFrames bool
}

var _ Handler = (*TimingHandler)(nil)

// NewTimingHandler builds a TimingHandler with Inner = NewCombiningHandler(inners...).
// Enabled is true when rec is non-nil; set Enabled explicitly if you attach Recorder later.
func NewTimingHandler(rec SpanRecorder, inners ...Handler) *TimingHandler {
	th := &TimingHandler{
		Recorder: rec,
		Inner:    NewCombiningHandler(inners...),
		Now:      time.Now,
		Enabled:  rec != nil,
	}
	return th
}

// resolveTraceContext returns the context to pass to Inner and Record (possibly enriched with
// *traceFrames), and *traceFrames when present or auto-installed. Callers: ctx, tf := t.resolveTraceContext(ctx).
func (t *TimingHandler) resolveTraceContext(ctx context.Context) (context.Context, *traceFrames) {
	if t == nil {
		return ctx, nil
	}
	if tf := traceFramesFromContext(ctx); tf != nil {
		return ctx, tf
	}
	if t.AutoEnsureTraceFrames {
		tf := newTraceFrames()
		return context.WithValue(ctx, traceFramesKey, tf), tf
	}
	return ctx, nil
}

func (t *TimingHandler) HandleText(ctx context.Context, text string) {
	if t.invokeInner() {
		defer t.Inner.HandleText(ctx, text)
	}

	if !t.active() {
		return
	}

	ts := t.now()
	t.record(ctx, SpanEvent{
		Name:    "text",
		Op:      SpanOpInstant,
		StartAt: ts,
		EndAt:   ts,
		Attrs:   map[string]string{"len": strconv.Itoa(len(text))},
	})
}

func (t *TimingHandler) HandleLLMStart(ctx context.Context, prompts []string) {
	if t.invokeInner() {
		defer t.Inner.HandleLLMStart(ctx, prompts)
	}

	if !t.active() {
		return
	}

	ts := t.now()
	t.record(ctx, SpanEvent{
		Name:    "llm_legacy",
		Op:      SpanOpInstant,
		StartAt: ts,
		EndAt:   ts,
		Attrs:   map[string]string{"prompts": strconv.Itoa(len(prompts))},
	})
}

func (t *TimingHandler) HandleLLMGenerateContentStart(ctx context.Context, ms []llms.MessageContent) {
	ctx, tf := t.resolveTraceContext(ctx)
	if t.invokeInner() {
		defer t.Inner.HandleLLMGenerateContentStart(ctx, ms)
	}

	if !t.active() {
		return
	}
	if tf == nil {
		return
	}

	tf.mu.Lock()
	defer tf.mu.Unlock()

	ts := t.now()
	tf.stack = append(tf.stack, stackFrame{kind: spanLLMGen, start: ts})
	tf.stream = streamAgg{}
	t.record(ctx, SpanEvent{
		Name:    "llm_generate",
		Op:      SpanOpStart,
		StartAt: ts,
		EndAt:   ts,
		Attrs:   map[string]string{"messages": strconv.Itoa(len(ms))},
	})
}

func (t *TimingHandler) HandleLLMGenerateContentEnd(ctx context.Context, res *llms.ContentResponse) {
	ctx, tf := t.resolveTraceContext(ctx)
	if t.invokeInner() {
		defer t.Inner.HandleLLMGenerateContentEnd(ctx, res)
	}

	if !t.active() {
		return
	}

	ts := t.now()
	if tf == nil {
		t.record(ctx, SpanEvent{
			Name:   "llm_generate",
			Op:     SpanOpEnd,
			EndAt:  ts,
			Orphan: true,
			Attrs:  attrsFromContentResponse(res),
		})
		return
	}

	tf.mu.Lock()
	defer tf.mu.Unlock()

	stack := &tf.stack
	stream := &tf.stream

	if len(*stack) == 0 || (*stack)[len(*stack)-1].kind != spanLLMGen {
		t.record(ctx, SpanEvent{
			Name:   "llm_generate",
			Op:     SpanOpEnd,
			EndAt:  ts,
			Orphan: true,
			Attrs:  attrsFromContentResponse(res),
		})
		return
	}

	fr := (*stack)[len(*stack)-1]
	*stack = (*stack)[:len(*stack)-1]
	dur := ts.Sub(fr.start)
	streamSnap := *stream
	*stream = streamAgg{}
	attrs := attrsFromContentResponse(res)
	attrs = mergeAttrs(attrs, finalizeStreamAttrs(streamSnap, ts, fr.start))
	t.record(ctx, SpanEvent{
		Name:     "llm_generate",
		Op:       SpanOpEnd,
		StartAt:  fr.start,
		EndAt:    ts,
		Duration: dur,
		Attrs:    attrs,
	})
}

func (t *TimingHandler) HandleLLMError(ctx context.Context, err error) {
	ctx, tf := t.resolveTraceContext(ctx)
	if t.invokeInner() {
		defer t.Inner.HandleLLMError(ctx, err)
	}

	if !t.active() {
		return
	}

	ts := t.now()
	if tf == nil {
		t.record(ctx, SpanEvent{
			Name:   "llm_generate",
			Op:     SpanOpError,
			EndAt:  ts,
			Err:    err,
			Orphan: true,
		})
		return
	}

	tf.mu.Lock()
	defer tf.mu.Unlock()

	stack := &tf.stack
	stream := &tf.stream

	if len(*stack) == 0 || (*stack)[len(*stack)-1].kind != spanLLMGen {
		t.record(ctx, SpanEvent{
			Name:   "llm_generate",
			Op:     SpanOpError,
			EndAt:  ts,
			Err:    err,
			Orphan: true,
		})
		return
	}

	fr := (*stack)[len(*stack)-1]
	*stack = (*stack)[:len(*stack)-1]
	dur := ts.Sub(fr.start)
	streamSnap := *stream
	*stream = streamAgg{}
	attrs := mergeAttrs(map[string]string{}, finalizeStreamAttrs(streamSnap, ts, fr.start))
	t.record(ctx, SpanEvent{
		Name:     "llm_generate",
		Op:       SpanOpError,
		StartAt:  fr.start,
		EndAt:    ts,
		Duration: dur,
		Err:      err,
		Attrs:    attrs,
	})
}

func (t *TimingHandler) HandleChainStart(ctx context.Context, inputs map[string]any) {
	ctx, tf := t.resolveTraceContext(ctx)
	if t.invokeInner() {
		defer t.Inner.HandleChainStart(ctx, inputs)
	}

	if !t.active() {
		return
	}
	if tf == nil {
		return
	}

	tf.mu.Lock()
	defer tf.mu.Unlock()

	ts := t.now()
	tf.stack = append(tf.stack, stackFrame{kind: spanChain, start: ts})
	t.record(ctx, SpanEvent{
		Name:    "chain",
		Op:      SpanOpStart,
		StartAt: ts,
		EndAt:   ts,
		Attrs:   map[string]string{"input_keys": strconv.Itoa(len(inputs))},
	})
}

func (t *TimingHandler) HandleChainEnd(ctx context.Context, outputs map[string]any) {
	ctx, tf := t.resolveTraceContext(ctx)
	if t.invokeInner() {
		defer t.Inner.HandleChainEnd(ctx, outputs)
	}

	if !t.active() {
		return
	}

	ts := t.now()
	if tf == nil {
		t.record(ctx, SpanEvent{
			Name:   "chain",
			Op:     SpanOpEnd,
			EndAt:  ts,
			Orphan: true,
			Attrs:  map[string]string{"output_keys": strconv.Itoa(len(outputs))},
		})
		return
	}

	tf.mu.Lock()
	defer tf.mu.Unlock()

	stack := &tf.stack

	if len(*stack) == 0 || (*stack)[len(*stack)-1].kind != spanChain {
		t.record(ctx, SpanEvent{
			Name:   "chain",
			Op:     SpanOpEnd,
			EndAt:  ts,
			Orphan: true,
			Attrs:  map[string]string{"output_keys": strconv.Itoa(len(outputs))},
		})
		return
	}

	fr := (*stack)[len(*stack)-1]
	*stack = (*stack)[:len(*stack)-1]
	t.record(ctx, SpanEvent{
		Name:     "chain",
		Op:       SpanOpEnd,
		StartAt:  fr.start,
		EndAt:    ts,
		Duration: ts.Sub(fr.start),
		Attrs:    map[string]string{"output_keys": strconv.Itoa(len(outputs))},
	})
}

func (t *TimingHandler) HandleChainError(ctx context.Context, err error) {
	ctx, tf := t.resolveTraceContext(ctx)
	if t.invokeInner() {
		defer t.Inner.HandleChainError(ctx, err)
	}

	if !t.active() {
		return
	}

	ts := t.now()
	if tf == nil {
		t.record(ctx, SpanEvent{
			Name:   "chain",
			Op:     SpanOpError,
			EndAt:  ts,
			Err:    err,
			Orphan: true,
		})
		return
	}

	tf.mu.Lock()
	defer tf.mu.Unlock()

	stack := &tf.stack

	if len(*stack) == 0 || (*stack)[len(*stack)-1].kind != spanChain {
		t.record(ctx, SpanEvent{
			Name:   "chain",
			Op:     SpanOpError,
			EndAt:  ts,
			Err:    err,
			Orphan: true,
		})
		return
	}

	fr := (*stack)[len(*stack)-1]
	*stack = (*stack)[:len(*stack)-1]
	t.record(ctx, SpanEvent{
		Name:     "chain",
		Op:       SpanOpError,
		StartAt:  fr.start,
		EndAt:    ts,
		Duration: ts.Sub(fr.start),
		Err:      err,
	})
}

func (t *TimingHandler) HandleToolStart(ctx context.Context, input string) {
	ctx, tf := t.resolveTraceContext(ctx)
	if t.invokeInner() {
		defer t.Inner.HandleToolStart(ctx, input)
	}

	if !t.active() {
		return
	}
	if tf == nil {
		return
	}

	tf.mu.Lock()
	defer tf.mu.Unlock()

	ts := t.now()
	tf.stack = append(tf.stack, stackFrame{kind: spanTool, start: ts})
	t.record(ctx, SpanEvent{
		Name:    "tool",
		Op:      SpanOpStart,
		StartAt: ts,
		EndAt:   ts,
		Attrs:   map[string]string{"input_len": strconv.Itoa(len(input))},
	})
}

func (t *TimingHandler) HandleToolEnd(ctx context.Context, output string) {
	ctx, tf := t.resolveTraceContext(ctx)
	if t.invokeInner() {
		defer t.Inner.HandleToolEnd(ctx, output)
	}

	if !t.active() {
		return
	}

	ts := t.now()
	if tf == nil {
		t.record(ctx, SpanEvent{
			Name:   "tool",
			Op:     SpanOpEnd,
			EndAt:  ts,
			Orphan: true,
			Attrs:  map[string]string{"output_len": strconv.Itoa(len(output))},
		})
		return
	}

	tf.mu.Lock()
	defer tf.mu.Unlock()

	stack := &tf.stack

	if len(*stack) == 0 || (*stack)[len(*stack)-1].kind != spanTool {
		t.record(ctx, SpanEvent{
			Name:   "tool",
			Op:     SpanOpEnd,
			EndAt:  ts,
			Orphan: true,
			Attrs:  map[string]string{"output_len": strconv.Itoa(len(output))},
		})
		return
	}

	fr := (*stack)[len(*stack)-1]
	*stack = (*stack)[:len(*stack)-1]
	t.record(ctx, SpanEvent{
		Name:     "tool",
		Op:       SpanOpEnd,
		StartAt:  fr.start,
		EndAt:    ts,
		Duration: ts.Sub(fr.start),
		Attrs:    map[string]string{"output_len": strconv.Itoa(len(output))},
	})
}

func (t *TimingHandler) HandleToolError(ctx context.Context, err error) {
	ctx, tf := t.resolveTraceContext(ctx)
	if t.invokeInner() {
		defer t.Inner.HandleToolError(ctx, err)
	}

	if !t.active() {
		return
	}

	ts := t.now()
	if tf == nil {
		t.record(ctx, SpanEvent{
			Name:   "tool",
			Op:     SpanOpError,
			EndAt:  ts,
			Err:    err,
			Orphan: true,
		})
		return
	}

	tf.mu.Lock()
	defer tf.mu.Unlock()

	stack := &tf.stack

	if len(*stack) == 0 || (*stack)[len(*stack)-1].kind != spanTool {
		t.record(ctx, SpanEvent{
			Name:   "tool",
			Op:     SpanOpError,
			EndAt:  ts,
			Err:    err,
			Orphan: true,
		})
		return
	}

	fr := (*stack)[len(*stack)-1]
	*stack = (*stack)[:len(*stack)-1]
	t.record(ctx, SpanEvent{
		Name:     "tool",
		Op:       SpanOpError,
		StartAt:  fr.start,
		EndAt:    ts,
		Duration: ts.Sub(fr.start),
		Err:      err,
	})
}

func (t *TimingHandler) HandleAgentAction(ctx context.Context, action schema.AgentAction) {
	if t.invokeInner() {
		defer t.Inner.HandleAgentAction(ctx, action)
	}

	if !t.active() {
		return
	}

	ts := t.now()
	t.record(ctx, SpanEvent{
		Name:    "agent_action",
		Op:      SpanOpInstant,
		StartAt: ts,
		EndAt:   ts,
		Attrs: map[string]string{
			"tool": action.Tool,
		},
	})
}

func (t *TimingHandler) HandleAgentFinish(ctx context.Context, finish schema.AgentFinish) {
	if t.invokeInner() {
		defer t.Inner.HandleAgentFinish(ctx, finish)
	}

	if !t.active() {
		return
	}

	ts := t.now()
	t.record(ctx, SpanEvent{
		Name:    "agent_finish",
		Op:      SpanOpInstant,
		StartAt: ts,
		EndAt:   ts,
		Attrs: map[string]string{
			"return_keys": strconv.Itoa(len(finish.ReturnValues)),
		},
	})
}

func (t *TimingHandler) HandleRetrieverStart(ctx context.Context, query string) {
	ctx, tf := t.resolveTraceContext(ctx)
	if t.invokeInner() {
		defer t.Inner.HandleRetrieverStart(ctx, query)
	}

	if !t.active() {
		return
	}
	if tf == nil {
		return
	}

	tf.mu.Lock()
	defer tf.mu.Unlock()

	ts := t.now()
	tf.stack = append(tf.stack, stackFrame{kind: spanRetriever, start: ts, query: query})
	t.record(ctx, SpanEvent{
		Name:    "retriever",
		Op:      SpanOpStart,
		StartAt: ts,
		EndAt:   ts,
		Attrs:   map[string]string{"query_len": strconv.Itoa(len(query))},
	})
}

func (t *TimingHandler) HandleRetrieverEnd(ctx context.Context, query string, documents []schema.Document) {
	ctx, tf := t.resolveTraceContext(ctx)
	if t.invokeInner() {
		defer t.Inner.HandleRetrieverEnd(ctx, query, documents)
	}

	if !t.active() {
		return
	}

	ts := t.now()
	if tf == nil {
		t.record(ctx, SpanEvent{
			Name:   "retriever",
			Op:     SpanOpEnd,
			EndAt:  ts,
			Orphan: true,
			Attrs: map[string]string{
				"docs": strconv.Itoa(len(documents)),
			},
		})
		return
	}

	tf.mu.Lock()
	defer tf.mu.Unlock()

	stack := &tf.stack

	if len(*stack) == 0 || (*stack)[len(*stack)-1].kind != spanRetriever {
		t.record(ctx, SpanEvent{
			Name:   "retriever",
			Op:     SpanOpEnd,
			EndAt:  ts,
			Orphan: true,
			Attrs: map[string]string{
				"docs": strconv.Itoa(len(documents)),
			},
		})
		return
	}

	fr := (*stack)[len(*stack)-1]
	*stack = (*stack)[:len(*stack)-1]
	attrs := map[string]string{
		"docs":      strconv.Itoa(len(documents)),
		"query_len": strconv.Itoa(len(query)),
	}

	if fr.query != query {
		attrs["query_mismatch"] = "true"
	}

	t.record(ctx, SpanEvent{
		Name:     "retriever",
		Op:       SpanOpEnd,
		StartAt:  fr.start,
		EndAt:    ts,
		Duration: ts.Sub(fr.start),
		Attrs:    attrs,
	})
}

func (t *TimingHandler) HandleStreamingFunc(ctx context.Context, chunk []byte) {
	ctx, tf := t.resolveTraceContext(ctx)
	if t.invokeInner() {
		defer t.Inner.HandleStreamingFunc(ctx, chunk)
	}

	if !t.active() {
		return
	}
	if tf == nil {
		return
	}

	tf.mu.Lock()
	defer tf.mu.Unlock()

	stack := &tf.stack
	stream := &tf.stream

	ts := t.now()
	n := len(chunk)
	var interArrival time.Duration
	if stream.chunkCount > 0 {
		interArrival = ts.Sub(stream.lastChunkAt)
	}

	stream.chunkCount++
	stream.bytesTotal += n
	stream.lastChunkAt = ts
	if stream.firstChunkAt == nil {
		stream.firstChunkAt = &ts
	}

	topLLM := len(*stack) > 0 && (*stack)[len(*stack)-1].kind == spanLLMGen
	orphan := !topLLM
	attrs := map[string]string{
		"bytes":            strconv.Itoa(n),
		"chunk_index":      strconv.Itoa(stream.chunkCount),
		"bytes_total":      strconv.Itoa(stream.bytesTotal),
		"inter_arrival_ns": strconv.FormatInt(interArrival.Nanoseconds(), 10),
	}

	if orphan {
		attrs["orphan_chunk"] = "true"
	}

	t.record(ctx, SpanEvent{
		Name:    "llm_stream",
		Op:      SpanOpStreamChunk,
		StartAt: ts,
		EndAt:   ts,
		Attrs:   attrs,
		Orphan:  orphan,
	})
}

func (t *TimingHandler) active() bool {
	return t != nil && t.Enabled && t.Recorder != nil
}

func finalizeStreamAttrs(s streamAgg, endTime, llmStart time.Time) map[string]string {
	out := map[string]string{
		"stream_chunks": strconv.Itoa(s.chunkCount),
		"stream_bytes":  strconv.Itoa(s.bytesTotal),
	}

	if s.firstChunkAt != nil {
		out["ttft_ns"] = strconv.FormatInt(s.firstChunkAt.Sub(llmStart).Nanoseconds(), 10)
		out["stream_duration_ns"] = strconv.FormatInt(endTime.Sub(*s.firstChunkAt).Nanoseconds(), 10)
	}

	return out
}

// invokeInner reports whether Inner is non-nil and should receive the deferred callback.
// Use with: if t.invokeInner() { defer t.Inner.HandleFoo(...) }.
func (t *TimingHandler) invokeInner() bool {
	return t != nil && t.Inner != nil
}

func (t *TimingHandler) now() time.Time {
	if t != nil && t.Now != nil {
		return t.Now()
	}
	return time.Now()
}

func (t *TimingHandler) record(ctx context.Context, e SpanEvent) {
	if t.Recorder != nil {
		t.Recorder.Record(ctx, e)
	}
}

func attrsFromContentResponse(res *llms.ContentResponse) map[string]string {
	if res == nil {
		return nil
	}

	out := make(map[string]string)
	for _, ch := range res.Choices {
		if ch == nil {
			continue
		}
		for k, v := range ch.GenerationInfo {
			s := stringifyGenInfo(v)
			out["geninfo_"+k] = s
			if canon, ok := canonicalUsageKey(k); ok && s != "" {
				if out[canon] == "" {
					out[canon] = s
				}
			}
		}
	}

	fillDerivedTotalTokens(out)

	if len(out) == 0 {
		return nil
	}

	return out
}

// canonicalUsageKey maps provider-specific GenerationInfo keys to stable span attribute names.
func canonicalUsageKey(key string) (string, bool) {
	switch {
	case strings.EqualFold(key, "prompt_tokens"),
		strings.EqualFold(key, "input_tokens"):
		return "prompt_tokens", true
	case strings.EqualFold(key, "completion_tokens"),
		strings.EqualFold(key, "output_tokens"):
		return "completion_tokens", true
	case strings.EqualFold(key, "total_tokens"):
		return "total_tokens", true
	case key == "PromptTokens":
		return "prompt_tokens", true
	case key == "CompletionTokens":
		return "completion_tokens", true
	case key == "TotalTokens":
		return "total_tokens", true
	default:
		return "", false
	}
}

func fillDerivedTotalTokens(attrs map[string]string) {
	if attrs == nil || attrs["total_tokens"] != "" {
		return
	}
	prompt := attrs["prompt_tokens"]
	comp := attrs["completion_tokens"]
	if prompt == "" || comp == "" {
		return
	}
	pi, err1 := strconv.ParseInt(prompt, 10, 64)
	ci, err2 := strconv.ParseInt(comp, 10, 64)
	if err1 != nil || err2 != nil {
		return
	}
	attrs["total_tokens"] = strconv.FormatInt(pi+ci, 10)
}

const stringifyGenInfoMaxLen = 4096

func stringifyGenInfo(v any) string {
	s := stringifyGenInfoCore(v)
	if len(s) <= stringifyGenInfoMaxLen {
		return s
	}
	return s[:stringifyGenInfoMaxLen-3] + "..."
}

func mergeAttrs(a, b map[string]string) map[string]string {
	if len(a) == 0 {
		return b
	}

	if len(b) == 0 {
		return a
	}

	for k, v := range b {
		a[k] = v
	}

	return a
}

func stringifyGenInfoCore(v any) string {
	switch x := v.(type) {
	case string, int, int8, int16, int32, int64, uint, uint8, uint16, uint32, uint64, float32, float64, bool:
		return fmt.Sprintf("%v", x)
	case json.Number:
		return x.String()
	default:
		b, err := json.Marshal(v)
		if err == nil {
			return string(b)
		}
		return fmt.Sprint(v)
	}
}
