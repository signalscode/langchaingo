package callbacks

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/tmc/langchaingo/llms"
)

func TestLLMGenerateSpy_Counts(t *testing.T) {
	t.Parallel()
	spy := &LLMGenerateSpy{}
	ctx := context.Background()
	spy.HandleLLMGenerateContentStart(ctx, nil)
	spy.HandleLLMGenerateContentEnd(ctx, &llms.ContentResponse{})
	spy.HandleLLMGenerateContentStart(ctx, []llms.MessageContent{{Role: llms.ChatMessageTypeHuman}})
	starts, ends := spy.Counts()
	assert.Equal(t, 2, starts)
	assert.Equal(t, 1, ends)
}
