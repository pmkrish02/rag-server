FROM golang:1.25-alpine
WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download
COPY practice.go .
RUN go build -o rag-server practice.go

FROM alpine:latest
WORKDIR /app
COPY --from=0 /app/rag-server /app/rag-server
EXPOSE 8000
CMD ["/app/rag-server"]