// Context callbacks example: attach TimingHandler + LogHandler via [callbacks.ContextWithHandler]
// so metrics and logs flow through [callbacks.EffectiveHandler] at library choke points
// (chains, agents, retrievers) without setting [callbacks.Handler] on every struct field.
//
// One SliceRecorder, TimingHandler, and context are reused for all demo sections; [callbacks.SliceRecorder.Reset]
// clears events between sections.
//
// Compare with examples/callbacks-metrics-log-example, which wires the same handlers explicitly
// on models, chains, tools, and executors.
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/tmc/langchaingo/agents"
	"github.com/tmc/langchaingo/callbacks"
	"github.com/tmc/langchaingo/chains"
	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/llms/fake"
	"github.com/tmc/langchaingo/prompts"
	"github.com/tmc/langchaingo/schema"
	"github.com/tmc/langchaingo/tools"
	"github.com/tmc/langchaingo/vectorstores"
)

func main() {
	// One logger + timing handler set in the Context; [SliceRecorder.Reset] clears events between runs.
	var (
		logging = callbacks.LogHandler{}
		rec     = &callbacks.SliceRecorder{}
		timing  = callbacks.NewTimingHandler(rec, logging)
		ctx     = callbacks.ContextWithHandler(context.Background(), timing)
		prompt  = prompts.NewPromptTemplate(
			"Reply with one short sentence. Task: {{.task}}",
			[]string{"task"},
		)
	)

	// 1. Direct model
	model := NewFakeLLM([]string{"The answer is 4."})

	fmt.Println("=== 1. direct model (context-only wiring) ===")
	_, err := llms.GenerateFromSinglePrompt(ctx, model,
		"Reply with one short sentence. Task: what is 2+2?",
	)
	if err != nil {
		log.Fatal(err)
	}
	printMetrics(rec)
	rec.Reset()

	// 2. Chain — [chains.Call]
	model = NewFakeLLM([]string{"The answer is 5."})
	chain := chains.NewLLMChain(model, prompt)

	fmt.Println("=== 2. chain + model (context-only wiring) ===")
	out, err := chains.Call(ctx, chain, map[string]any{"task": "what is 2+3?"})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("chain output:", out["text"])
	printMetrics(rec)
	rec.Reset()

	// 3. Retriever — [vectorstores.Retriever.GetRelevantDocuments]
	store := staticResultsVS{docs: []schema.Document{{PageContent: "Paris is the capital of France."}}}
	r := vectorstores.ToRetriever(store, 1)

	fmt.Println("=== 3. retriever ===")
	docs, err := r.GetRelevantDocuments(ctx, "capital of France")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("retrieved:", docs[0].PageContent)
	printMetrics(rec)
	rec.Reset()

	// 4. Agent + executor — tool and LLM lifecycle use context
	agentModel := NewFakeLLM([]string{
		"Action: calculator\nAction Input: 8*8",
		"Final Answer: 64",
	})
	agent := agents.NewOneShotAgent(
		agentModel,
		[]tools.Tool{tools.Calculator{}},
	)
	executor := agents.NewExecutor(
		agent,
		agents.WithMaxIterations(3),
	)

	fmt.Println("=== 4. agent + tool + model (context-only wiring) ===")
	agentOut, err := executor.Call(ctx, map[string]any{"input": "What is 8*8?"})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("agent output:", agentOut["output"])
	printMetrics(rec)

	fmt.Println("**Note:** direct [tools.Tool] calls do not merge [callbacks.ContextWithHandler] in the library; use an agent/executor or wrap the tool.")
}

func printMetrics(rec *callbacks.SliceRecorder) {
	fmt.Println()
	fmt.Println("=== metrics (SliceRecorder / SpanEvent) ===")
	for i, e := range rec.Events {
		line := fmt.Sprintf("%2d  %-14s %-8s", i, e.Name, e.Op)
		if e.Duration > 0 {
			line += fmt.Sprintf("  duration=%s", e.Duration)
		}
		if e.Orphan {
			line += "  orphan=true"
		}
		if len(e.Attrs) > 0 {
			line += fmt.Sprintf("  attrs=%v", e.Attrs)
		}
		fmt.Println(line)
	}
	fmt.Println()
}

///// ------------------------------------------------------------------------------------------------
///// Code below here is for mocking fake LLM and is not required in normal usage
///// ------------------------------------------------------------------------------------------------

// llmCallbacks wraps a model and invokes HandleLLM* via [callbacks.EffectiveHandler], like providers
// that merge context with struct-level handlers. The fake LLM does not invoke callbacks by itself.
type llmCallbacks struct {
	inner   llms.Model
	handler callbacks.Handler
}

// NewFakeLLM builds a model that observes callbacks from context (and optional field handlers).
func NewFakeLLM(responses []string, handlers ...callbacks.Handler) *llmCallbacks {
	handler := callbacks.Coalesce(handlers...)
	return &llmCallbacks{inner: fake.NewFakeLLM(responses), handler: handler}
}

func (m *llmCallbacks) GenerateContent(ctx context.Context, messages []llms.MessageContent, options ...llms.CallOption) (*llms.ContentResponse, error) {
	h := callbacks.EffectiveHandler(ctx, m.handler)
	if h != nil {
		h.HandleLLMGenerateContentStart(ctx, messages)
	}
	resp, err := m.inner.GenerateContent(ctx, messages, options...)
	if err == nil && resp != nil {
		withMockGenerationInfo(resp, messages)
	}
	if h != nil {
		if err != nil {
			h.HandleLLMError(ctx, err)
		} else {
			h.HandleLLMGenerateContentEnd(ctx, resp)
		}
	}
	return resp, err
}

func withMockGenerationInfo(res *llms.ContentResponse, messages []llms.MessageContent) {
	if res == nil || len(res.Choices) == 0 || res.Choices[0] == nil {
		return
	}
	ch := res.Choices[0]
	promptChars := 0
	for _, msg := range messages {
		for _, part := range msg.Parts {
			switch t := part.(type) {
			case llms.TextContent:
				promptChars += len(t.Text)
			default:
				promptChars += len(fmt.Sprint(part))
			}
		}
	}
	completionChars := len(ch.Content)
	ch.GenerationInfo = map[string]any{
		"prompt_tokens":     promptChars,
		"completion_tokens": completionChars,
		"total_tokens":      promptChars + completionChars,
	}
}

func (m *llmCallbacks) Call(ctx context.Context, prompt string, options ...llms.CallOption) (string, error) {
	return llms.GenerateFromSinglePrompt(ctx, m, prompt, options...)
}

// staticResultsVS is a minimal [vectorstores.VectorStore] for the example.
type staticResultsVS struct {
	docs []schema.Document
}

func (s staticResultsVS) AddDocuments(ctx context.Context, docs []schema.Document, options ...vectorstores.Option) ([]string, error) {
	return nil, nil
}

func (s staticResultsVS) SimilaritySearch(ctx context.Context, query string, numDocuments int, options ...vectorstores.Option) ([]schema.Document, error) {
	if len(s.docs) == 0 {
		return nil, nil
	}
	n := numDocuments
	if n > len(s.docs) {
		n = len(s.docs)
	}
	return s.docs[:n], nil
}
