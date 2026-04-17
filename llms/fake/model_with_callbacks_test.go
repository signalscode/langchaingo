package fake

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/tmc/langchaingo/callbacks"
	"github.com/tmc/langchaingo/llms"
)

func TestModelWithCallbacks_invokesCallbacks(t *testing.T) {
	t.Parallel()
	spy := &callbacks.LLMGenerateSpy{}
	m := NewModelWithCallbacks([]string{"hi"}, spy)
	ctx := context.Background()
	_, err := m.GenerateContent(ctx, []llms.MessageContent{
		{Role: llms.ChatMessageTypeHuman, Parts: []llms.ContentPart{llms.TextContent{Text: "x"}}},
	})
	require.NoError(t, err)
	starts, ends := spy.Counts()
	assert.Equal(t, 1, starts)
	assert.Equal(t, 1, ends)
}

func TestModelWithCallbacks_EffectiveHandlerMergesContext(t *testing.T) {
	t.Parallel()
	fieldSpy := &callbacks.LLMGenerateSpy{}
	ctxSpy := &callbacks.LLMGenerateSpy{}
	ctx := callbacks.ContextWithHandler(context.Background(), ctxSpy)
	m := NewModelWithCallbacks([]string{"ok"}, fieldSpy)
	_, err := m.GenerateContent(ctx, []llms.MessageContent{
		{Role: llms.ChatMessageTypeHuman, Parts: []llms.ContentPart{llms.TextContent{Text: "y"}}},
	})
	require.NoError(t, err)
	fs, fe := fieldSpy.Counts()
	cs, ce := ctxSpy.Counts()
	assert.Equal(t, 1, fs)
	assert.Equal(t, 1, fe)
	assert.Equal(t, 1, cs)
	assert.Equal(t, 1, ce)
}
