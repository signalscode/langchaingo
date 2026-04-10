package callbacks

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestSpanEvent_MarshalJSON_minimal(t *testing.T) {
	t.Parallel()
	b, err := json.Marshal(SpanEvent{Name: "chain", Op: SpanOpStart})
	require.NoError(t, err)
	assert.JSONEq(t, `{"name":"chain","op":"start"}`, string(b))
}

func TestSpanEvent_MarshalJSON_full(t *testing.T) {
	t.Parallel()
	start := time.Date(2024, 3, 15, 12, 30, 0, 123456789, time.FixedZone("CST", -6*3600))
	end := start.Add(2 * time.Second)
	e := SpanEvent{
		Name:     "llm_generate",
		Op:       SpanOpEnd,
		StartAt:  start,
		EndAt:    end,
		Duration: 2 * time.Second,
		Err:      errors.New("boom"),
		Attrs:    map[string]string{"model": "x"},
		Orphan:   true,
	}
	b, err := json.Marshal(e)
	require.NoError(t, err)

	var m map[string]any
	require.NoError(t, json.Unmarshal(b, &m))
	assert.Equal(t, "llm_generate", m["name"])
	assert.Equal(t, "end", m["op"])
	assert.Equal(t, start.UTC().Format(time.RFC3339Nano), m["start_at"])
	assert.Equal(t, end.UTC().Format(time.RFC3339Nano), m["end_at"])
	assert.Equal(t, (2 * time.Second).String(), m["duration"])
	assert.Equal(t, "boom", m["err"])
	assert.Equal(t, map[string]any{"model": "x"}, m["attrs"])
	assert.Equal(t, true, m["orphan"])
}

func TestSpanEvent_MarshalJSON_omitsZero(t *testing.T) {
	t.Parallel()
	e := SpanEvent{Name: "x", Op: SpanOpInstant}
	b, err := json.Marshal(e)
	require.NoError(t, err)
	s := string(b)
	for _, key := range []string{"start_at", "end_at", "duration", "err", "attrs", "orphan"} {
		assert.False(t, strings.Contains(s, `"`+key+`"`), "should omit %q: %s", key, s)
	}
}

func TestSpanEvent_MarshalJSON_nilError(t *testing.T) {
	t.Parallel()
	b, err := json.Marshal(SpanEvent{Name: "a", Op: SpanOpEnd, Err: nil})
	require.NoError(t, err)
	assert.NotContains(t, string(b), `"err"`)
}

func TestMarshalSpanEvents(t *testing.T) {
	t.Parallel()
	events := []SpanEvent{
		{Name: "c", Op: SpanOpStart},
		{Name: "c", Op: SpanOpEnd, Duration: time.Millisecond},
	}
	b, err := MarshalSpanEvents(events)
	require.NoError(t, err)
	assert.JSONEq(t, `[{"name":"c","op":"start"},{"name":"c","op":"end","duration":"1ms"}]`, string(b))
}

func TestSpanEventsMetadataJSON(t *testing.T) {
	t.Parallel()
	raw, err := SpanEventsMetadataJSON([]SpanEvent{{Name: "t", Op: SpanOpInstant}})
	require.NoError(t, err)
	assert.JSONEq(t, `[{"name":"t","op":"instant"}]`, string(raw))
}
