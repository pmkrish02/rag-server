package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strings"
	"os"
	"context"

	"github.com/weaviate/weaviate-go-client/v5/weaviate"
	"github.com/weaviate/weaviate-go-client/v5/weaviate/auth"
	"github.com/weaviate/weaviate-go-client/v5/weaviate/graphql"
	"github.com/weaviate/weaviate/entities/models"
	"google.golang.org/genai"
)

type ragServer struct {
	ctx         context.Context
	wvClient    *weaviate.Client
	genaiClient *genai.Client
}

func (rs *ragServer) documentHandler(w http.ResponseWriter, r *http.Request) {

	type document struct {
		Text string
	}
	type addRequest struct {
		Documents []document
	}
	var docreq addRequest

	err := json.NewDecoder(r.Body).Decode(&docreq)
	if err != nil {
		http.Error(w, "Bad JSON", http.StatusBadRequest)
		return
	}

	contents := make([]*genai.Content, len(docreq.Documents))
	for i, doc := range docreq.Documents {
		contents[i] = genai.NewContentFromText(doc.Text, genai.RoleUser)
	}

	rsp, err := rs.genaiClient.Models.EmbedContent(rs.ctx, "gemini-embedding-004", contents, nil)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	if len(rsp.Embeddings) != len(docreq.Documents) {
		http.Error(w, "embedding count mismatch", http.StatusInternalServerError)
		return
	}

	objects := make([]*models.Object, len(docreq.Documents))
	for i, doc := range docreq.Documents {
		objects[i] = &models.Object{
			Class: "Document",
			Properties: map[string]any{
				"text": doc.Text,
			},
			Vector: rsp.Embeddings[i].Values,
		}
	}

	_, err = rs.wvClient.Batch().ObjectsBatcher().WithObjects(objects...).Do(rs.ctx)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	fmt.Fprintf(w, "stored %d documents", len(docreq.Documents))

}

func (rs *ragServer) queryHandler(w http.ResponseWriter, r *http.Request) {
	type Queryrequest struct {
		Querycontent string `json:"querycontent"`
	}
	var qr Queryrequest
	err := json.NewDecoder(r.Body).Decode(&qr)
	if err != nil {
		http.Error(w, "BAD JSON", http.StatusBadRequest)
		return
	}

	contents := []*genai.Content{
		genai.NewContentFromText(qr.Querycontent, genai.RoleUser),
	}
	rsp, err := rs.genaiClient.Models.EmbedContent(rs.ctx, "gemini-embedding-004", contents, nil)

	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	gql := rs.wvClient.GraphQL()
	result, err := gql.Get().
		WithNearVector(
			gql.NearVectorArgBuilder().WithVector(rsp.Embeddings[0].Values)).
		WithClassName("Document").
		WithFields(graphql.Field{Name: "text"}).
		WithLimit(3).
		Do(rs.ctx)

	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	contentTexts, err := decodeGetResults(result)
	if err != nil {
		http.Error(w, fmt.Errorf("reading weaviate response: %w", err).Error(), http.StatusInternalServerError)
		return
	}

	ragQuery := fmt.Sprintf("Based on this context:\n%s\n\nAnswer this question: %s", strings.Join(contentTexts, "\n"), qr.Querycontent)

	genResp, err := rs.genaiClient.Models.GenerateContent(rs.ctx, "gemini-3.1-flash-lite-preview", genai.Text(ragQuery), nil)
	if err != nil {
		log.Printf("calling generative model: %v", err.Error())
		http.Error(w, "generative model error", http.StatusInternalServerError)
		return
	}

	if len(genResp.Candidates) != 1 {
		log.Printf("got %v candidates, expected 1", len(genResp.Candidates))
		http.Error(w, "generative model error", http.StatusInternalServerError)
		return
	}

	js, err := json.Marshal(genResp.Text())
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.Write(js)

}
func (rs *ragServer)HealthHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "only GET allowed", http.StatusMethodNotAllowed)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

func decodeGetResults(result *models.GraphQLResponse) ([]string, error) {
	data, ok := result.Data["Get"]
	if !ok {
		return nil, fmt.Errorf("no Get key in result")
	}
	doc, ok := data.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("unexpected type")
	}
	slc, ok := doc["Document"].([]any)
	if !ok {
		return nil, fmt.Errorf("no Document results")
	}
	var out []string
	for _, s := range slc {
		smap, ok := s.(map[string]any)
		if !ok {
			continue
		}
		text, ok := smap["text"].(string)
		if ok {
			out = append(out, text)
		}
	}
	return out, nil
}

func main() {
	ctx := context.Background()
	cfg := weaviate.Config{
		Host:       os.Getenv("WEAVIATE_HOSTNAME"),
		Scheme:     "https",
		AuthConfig: auth.ApiKey{Value: os.Getenv("WEAVIATE_API_KEY")},
	}

	wvClient, err := weaviate.NewClient(cfg)
	if err != nil {
		fmt.Println(err)
	}
	genaiClient, err := genai.NewClient(ctx, nil)
	if err != nil {
		log.Fatal(err)
	}

	classObj := &models.Class{
		Class: "Document",
		Properties: []*models.Property{
			{Name: "text", DataType: []string{"text"}},
		},
	}
	exists, err := wvClient.Schema().ClassExistenceChecker().WithClassName("Document").Do(ctx)
	if err != nil {
		log.Fatal(err)
	}
	if !exists {
		err = wvClient.Schema().ClassCreator().WithClass(classObj).Do(ctx)
		if err != nil {
			log.Fatal(err)
		}
		fmt.Println("Document collection created")
	}

	server := &ragServer{
		ctx:         ctx,
		wvClient:    wvClient,
		genaiClient: genaiClient,
	}
	router := http.NewServeMux()
	router.HandleFunc("GET /health", server.HealthHandler)
	router.HandleFunc("POST /documents/", server.documentHandler)
	router.HandleFunc("POST /query/", server.queryHandler)
	fmt.Println("Rag serverrunning on :8000")
	http.ListenAndServe(":8000", router)
}
