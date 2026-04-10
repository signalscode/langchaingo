package callbacks

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestHandlerFromContext_nilContext(t *testing.T) {
	t.Parallel()
	h, ok := HandlerFromContext(context.TODO())
	assert.False(t, ok)
	assert.Nil(t, h)
}

func TestContextWithHandler_emptyReturnsUnchanged(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	out := ContextWithHandler(ctx)
	assert.Equal(t, ctx, out)
	out = ContextWithHandler(ctx, nil, nil)
	assert.Equal(t, ctx, out)
}

func TestContextWithHandler_composeWithExisting(t *testing.T) {
	t.Parallel()
	first := &SimpleHandler{}
	second := &SimpleHandler{}
	ctx := ContextWithHandler(context.Background(), first)
	ctx = ContextWithHandler(ctx, second)
	h, ok := HandlerFromContext(ctx)
	require.True(t, ok)
	ch, ok := h.(CombiningHandler)
	require.True(t, ok)
	require.Len(t, ch.Callbacks, 2)
	assert.Equal(t, first, ch.Callbacks[0])
	assert.Equal(t, second, ch.Callbacks[1])
}

func TestEffectiveHandler_order(t *testing.T) {
	t.Parallel()
	ctx := ContextWithHandler(context.Background(), &SimpleHandler{})
	field := &SimpleHandler{}
	merged := EffectiveHandler(ctx, field)
	ch, ok := merged.(CombiningHandler)
	require.True(t, ok)
	require.Len(t, ch.Callbacks, 2)
}
