package fake

import (
	"context"
	"fmt"

	"github.com/tmc/langchaingo/callbacks"
	"github.com/tmc/langchaingo/llms"
)

// ModelWithCallbacks wraps a scripted [LLM] and invokes LLM callback hooks the same way
// production providers do, merging optional struct-level handlers with context via
// [callbacks.EffectiveHandler].
type ModelWithCallbacks struct {
	inner    *LLM
	handlers callbacks.Handler
}

// NewModelWithCallbacks returns a [llms.Model] backed by [NewFakeLLM] that fires
// HandleLLMGenerateContentStart, HandleLLMGenerateContentEnd, and HandleLLMError on each
// [GenerateContent] call. Use this in examples and tests when you need callback/metrics behavior
// that the bare [LLM] does not implement on its own.
func NewModelWithCallbacks(responses []string, handlers ...callbacks.Handler) *ModelWithCallbacks {
	return &ModelWithCallbacks{
		inner:    NewFakeLLM(responses),
		handlers: callbacks.NewCombiningHandler(handlers...),
	}
}

// GenerateContent implements [llms.Model].
func (m *ModelWithCallbacks) GenerateContent(ctx context.Context, messages []llms.MessageContent, options ...llms.CallOption) (*llms.ContentResponse, error) {
	h := callbacks.EffectiveHandler(ctx, m.handlers)
	if h != nil {
		h.HandleLLMGenerateContentStart(ctx, messages)
	}
	resp, err := m.inner.GenerateContent(ctx, messages, options...)
	if err == nil && resp != nil {
		ApplyMockGenerationInfo(resp, messages)
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

// Call implements [llms.Model].
func (m *ModelWithCallbacks) Call(ctx context.Context, prompt string, options ...llms.CallOption) (string, error) {
	return llms.GenerateFromSinglePrompt(ctx, m, prompt, options...)
}

// ApplyMockGenerationInfo sets [llms.ContentChoice].GenerationInfo using character counts as
// stand-in token metrics so handlers like [callbacks.TimingHandler] can emit token attributes
// in demos and tests without a real tokenizer.
func ApplyMockGenerationInfo(res *llms.ContentResponse, messages []llms.MessageContent) {
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
