package openai

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/tmc/langchaingo/callbacks"
	"github.com/tmc/langchaingo/llms"
)

func TestGenerateContent_ContextHandlerWithoutStructField(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{
  "id": "chatcmpl-test",
  "object": "chat.completion",
  "created": 1,
  "model": "gpt-3.5-turbo",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "ok"},
    "finish_reason": "stop"
  }]
}`))
	}))
	t.Cleanup(server.Close)

	llm, err := New(
		WithToken("test-token"),
		WithBaseURL(server.URL),
		WithModel("gpt-3.5-turbo"),
	)
	require.NoError(t, err)
	assert.Nil(t, llm.CallbacksHandler)

	spy := &callbacks.LLMGenerateSpy{}
	ctx := callbacks.ContextWithHandler(context.Background(), spy)

	msgs := []llms.MessageContent{
		{Role: llms.ChatMessageTypeHuman, Parts: []llms.ContentPart{llms.TextPart("hello")}},
	}
	_, err = llm.GenerateContent(ctx, msgs)
	require.NoError(t, err)

	starts, ends := spy.Counts()
	assert.Equal(t, 1, starts, "context handler should see LLM start without struct CallbacksHandler")
	assert.Equal(t, 1, ends, "context handler should see LLM end without struct CallbacksHandler")
}
