package googleai

import (
	"context"

	genai "google.golang.org/genai"
)

// CreateEmbedding creates embeddings from texts using the new API.
func (g *GoogleAI) CreateEmbedding(ctx context.Context, texts []string) ([][]float32, error) {
	response, err := g.client.Models.EmbedContent(ctx, g.opts.DefaultModel, genai.Text(texts[0]), &genai.EmbedContentConfig{TaskType: "RETRIEVAL_QUERY"})
	if err != nil {
		return nil, err
	}
	return [][]float32{response.Embeddings[0].Values}, nil
}
