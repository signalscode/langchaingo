package callbacks

// Coalesce returns a single handler that invokes handlers in order (left to right).
// Nil handlers are dropped. If all are nil or the slice is empty, it returns nil.
// If exactly one non-nil handler remains, it returns that handler without allocating CombiningHandler.
func Coalesce(handlers ...Handler) Handler {
	out := make([]Handler, 0, len(handlers))
	for _, h := range handlers {
		if h != nil {
			out = append(out, h)
		}
	}
	switch len(out) {
	case 0:
		return nil
	case 1:
		return out[0]
	default:
		return CombiningHandler{Callbacks: out}
	}
}
