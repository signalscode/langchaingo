package callbacks

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestCoalesce(t *testing.T) {
	t.Parallel()

	t.Run("empty returns nil", func(t *testing.T) {
		require.Nil(t, Coalesce())
	})
	t.Run("all nil returns nil", func(t *testing.T) {
		require.Nil(t, Coalesce(nil, nil))
	})
	t.Run("first non-nil wins", func(t *testing.T) {
		one := SimpleHandler{}
		require.Equal(t, one, Coalesce(nil, one))
	})
	t.Run("two non-nil combines", func(t *testing.T) {
		a := &callCounter{}
		b := &callCounter{}
		c := Coalesce(a, b).(CombiningHandler)
		require.Len(t, c.Callbacks, 2)
	})
	t.Run("preserves left-to-right order", func(t *testing.T) {
		a := &callCounter{}
		b := &callCounter{}
		c := &callCounter{}
		merged, ok := Coalesce(a, nil, b, c).(CombiningHandler)
		require.True(t, ok)
		require.Equal(t, []Handler{a, b, c}, merged.Callbacks)
	})
}
