package cloudflare

import (
	"context"
	"net/http"
	"net/http/httptest"
	"net/url"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/tmc/langchaingo/callbacks"
	"github.com/tmc/langchaingo/llms"
)

func TestGenerateContent_ContextHandlerWithoutStructField(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		response := `{
			"result": { "response": "Hello!" },
			"success": true,
			"errors": [],
			"messages": []
		}`
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(response))
	}))
	t.Cleanup(server.Close)

	serverURL, err := url.Parse(server.URL)
	require.NoError(t, err)

	llm, err := New(
		WithToken("test-token"),
		WithAccountID("test-account-id"),
		WithModel("test-model"),
		WithCloudflareServerURL(serverURL),
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
	assert.Equal(t, 1, starts)
	assert.Equal(t, 1, ends)
}
