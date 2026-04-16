// Context callbacks example: attach TimingHandler + LogHandler via [callbacks.ContextWithHandler]
// so metrics and logs flow through [callbacks.EffectiveHandler] at library choke points
// (chains, agents, retrievers) without setting [callbacks.Handler] on every struct field.
//
// TimingHandler.AutoEnsureTraceFrames is enabled so the first stack-aware callback allocates
// *traceFrames on the request context; downstream code must pass derived contexts from that
// branch (see [callbacks.TimingHandler] and [callbacks.EnsureTraceFrames] for production).
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
		prompt  = prompts.NewPromptTemplate(
			"Reply with one short sentence. Task: {{.task}}",
			[]string{"task"},
		)
	)
	timing.AutoEnsureTraceFrames = true
	ctx := callbacks.ContextWithHandler(context.Background(), timing)

	// 1. Direct model
	model := fake.NewModelWithCallbacks([]string{"The answer is 4."})

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
	model = fake.NewModelWithCallbacks([]string{"The answer is 5."})
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
	agentModel := fake.NewModelWithCallbacks([]string{
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
