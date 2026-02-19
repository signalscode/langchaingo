package googleai

import (
	"context"
	"fmt"

	"google.golang.org/genai"
)

// CreateEmbedding creates embeddings from texts.
func (g *GoogleAI) CreateEmbedding(ctx context.Context, texts []string) ([][]float32, error) {
	results := make([][]float32, 0, len(texts))

	// The new SDK supports batching automatically within EmbedContent
	// Process in batches of 100 (API limit)
	batchSize := 100
	for i := 0; i < len(texts); i += batchSize {
		end := i + batchSize
		if end > len(texts) {
			end = len(texts)
		}
		batch := texts[i:end]

		// Build contents for the batch
		contents := make([]*genai.Content, 0, len(batch))
		for _, text := range batch {
			contents = append(contents, &genai.Content{
				Parts: []*genai.Part{genai.NewPartFromText(text)},
			})
		}

		// Call EmbedContent
		resp, err := g.client.Models.EmbedContent(ctx, g.opts.DefaultEmbeddingModel, contents, nil)
		if err != nil {
			return nil, fmt.Errorf("failed to create embeddings: %w", err)
		}

		if resp.Embeddings == nil {
			return nil, fmt.Errorf("no embeddings returned for batch")
		}

		// Extract embeddings from response
		for _, emb := range resp.Embeddings {
			results = append(results, emb.Values)
		}
	}

	return results, nil
}
