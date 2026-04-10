// Callbacks example: structured metrics (TimingHandler + SpanRecorder) and stdout logging (LogHandler).
//
// Wiring matches the recommended stack: TimingHandler outermost so span durations include inner work,
// with LogHandler as Inner so human-readable lines print after the timing hook runs its body.
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
	"github.com/tmc/langchaingo/tools"
)

func main() {
	ctx := context.Background()
	logging := callbacks.LogHandler{}
	prompt := prompts.NewPromptTemplate(
		"Reply with one short sentence. Task: {{.task}}",
		[]string{"task"},
	)

	// 1. Direct model call — handler is wired on the model only (no chain callbacks).
	rec := &callbacks.SliceRecorder{}
	timing := callbacks.NewTimingHandler(rec, logging)
	model := NewFakeLLM([]string{"The answer is 4."}, timing)

	fmt.Println("=== 1. direct model (stdout via LogHandler) ===")
	_, err := llms.GenerateFromSinglePrompt(ctx, model,
		"Reply with one short sentence. Task: what is 2+2?",
	)
	if err != nil {
		log.Fatal(err)
	}
	printMetrics(rec)

	// 2. Chain — register the same handler on the chain (WithCallback) and on the model so both
	// HandleChain* and HandleLLM* reach LogHandler.
	rec = &callbacks.SliceRecorder{}
	timing = callbacks.NewTimingHandler(rec, logging)
	model = NewFakeLLM([]string{"The answer is 5."}, timing)
	chain := chains.NewLLMChain(model, prompt, chains.WithCallback(timing))

	fmt.Println("=== 2. chain + model (chain and LLM logs) ===")
	out, err := chains.Call(ctx, chain, map[string]any{"task": "what is 2+3?"})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("chain output:", out["text"])
	printMetrics(rec)

	// 3. Direct tool call — useful when tools are used outside an agent loop.
	rec = &callbacks.SliceRecorder{}
	timing = callbacks.NewTimingHandler(rec, logging)
	calc := tools.Calculator{CallbacksHandler: timing}

	fmt.Println("=== 3. direct tool (tool logs + tool spans) ===")
	toolOut, err := calc.Call(ctx, "3*7")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("tool output:", toolOut)
	printMetrics(rec)

	// 4. Agent + tool + chain — one handler on agent/model gives full lifecycle:
	// chain start/end, llm start/end, agent action/finish, and tool start/end.
	rec = &callbacks.SliceRecorder{}
	timing = callbacks.NewTimingHandler(rec, logging)
	agentModel := NewFakeLLM([]string{
		"Action: calculator\nAction Input: 8*8",
		"Final Answer: 64",
	}, timing)
	agent := agents.NewOneShotAgent(
		agentModel,
		[]tools.Tool{tools.Calculator{}},
		agents.WithCallbacksHandler(timing),
	)
	executor := agents.NewExecutor(
		agent,
		agents.WithCallbacksHandler(timing),
		agents.WithMaxIterations(3),
	)

	fmt.Println("=== 4. agent + tool + model (full lifecycle logs) ===")
	agentOut, err := executor.Call(ctx, map[string]any{"input": "What is 8*8?"})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("agent output:", agentOut["output"])
	printMetrics(rec)

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

// llmCallbacks wraps a model and invokes HandleLLM* on the given handler, similar to provider
// implementations (e.g. OpenAI). The fake LLM does not do this by itself.
type llmCallbacks struct {
	inner   llms.Model
	handler callbacks.Handler
}

func NewFakeLLM(responses []string, handlers ...callbacks.Handler) *llmCallbacks {
	handler := callbacks.Coalesce(handlers...)
	return &llmCallbacks{inner: fake.NewFakeLLM(responses), handler: handler}
}

func (m *llmCallbacks) GenerateContent(ctx context.Context, messages []llms.MessageContent, options ...llms.CallOption) (*llms.ContentResponse, error) {
	h := m.handler
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

// withMockGenerationInfo sets ContentChoice.GenerationInfo using string lengths as stand-in token
// counts so TimingHandler can emit prompt_tokens / completion_tokens / total_tokens on llm_generate end.
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
				// Rough stand-in for non-text parts (images, tool payloads).
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
