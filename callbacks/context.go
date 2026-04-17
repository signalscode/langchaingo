package callbacks

import (
	"context"
)

type handlerCtxKey struct{}

// ContextWithHandler attaches handlers to ctx. If ctx already carries a handler, the new
// handlers are composed after the existing one (NewCombiningHandler(existing, NewCombiningHandler(handlers...))).
// If all handlers are nil, ctx is returned unchanged.
func ContextWithHandler(parent context.Context, handlers ...Handler) context.Context {
	h := NewCombiningHandler(handlers...)
	if h == nil {
		return parent
	}
	if prev, ok := HandlerFromContext(parent); ok && prev != nil {
		h = NewCombiningHandler(prev, h)
	}
	return context.WithValue(parent, handlerCtxKey{}, h)
}

// HandlerFromContext returns the handler stored on ctx, if any.
func HandlerFromContext(ctx context.Context) (Handler, bool) {
	if ctx == nil {
		return nil, false
	}
	v := ctx.Value(handlerCtxKey{})
	if v == nil {
		return nil, false
	}
	h, ok := v.(Handler)
	if !ok || h == nil {
		return nil, false
	}
	return h, true
}

// EffectiveHandler returns NewCombiningHandler(handlerFromContext, handlers...) with nils dropped.
// The context-derived handler runs first when present. A process-wide default may be
// composed here in a future version.
func EffectiveHandler(ctx context.Context, handlers ...Handler) Handler {
	ctxH, _ := HandlerFromContext(ctx)
	return NewCombiningHandler(append([]Handler{ctxH}, handlers...)...)
}
