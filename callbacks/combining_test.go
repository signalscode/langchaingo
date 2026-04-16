package callbacks

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestNewCombiningHandler(t *testing.T) {
	t.Parallel()

	t.Run("empty returns nil", func(t *testing.T) {
		require.Nil(t, NewCombiningHandler())
	})
	t.Run("all nil returns nil", func(t *testing.T) {
		require.Nil(t, NewCombiningHandler(nil, nil))
	})
	t.Run("single non-nil returned as-is", func(t *testing.T) {
		one := SimpleHandler{}
		require.Equal(t, one, NewCombiningHandler(nil, one))
	})
	t.Run("two non-nil combines", func(t *testing.T) {
		a := &callCounter{}
		b := &callCounter{}
		c := NewCombiningHandler(a, b).(CombiningHandler)
		require.Len(t, c.Callbacks, 2)
	})
	t.Run("preserves left-to-right order", func(t *testing.T) {
		a := &callCounter{}
		b := &callCounter{}
		c := &callCounter{}
		merged, ok := NewCombiningHandler(a, nil, b, c).(CombiningHandler)
		require.True(t, ok)
		require.Equal(t, []Handler{a, b, c}, merged.Callbacks)
	})
}
