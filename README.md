# Odyssey Hierarchical RAG System - Technical Documentation

## Project Overview

The Odyssey Hierarchical RAG System is a sophisticated Retrieval-Augmented Generation application built for analyzing and querying Homer's "The Odyssey." The system implements hierarchical semantic chunking with multi-level document processing, enabling context-aware question answering across different abstraction levels.

## Architecture Components

### Backend API (FastAPI)

- **Framework**: FastAPI with Python 3.8+
- **Core Pipeline**: `OdysseyHierarchicalPipeline` - Handles document processing, embedding generation, and hierarchical retrieval
- **Vector Database**: FAISS for efficient similarity search
- **Embedding Model**: Google Generative AI embeddings (via Google API)
- **LLM**: Google's Generative AI for response generation

### Frontend Interface (React/TypeScript)

- **Framework**: React 18 with TypeScript
- **UI Library**: Material-UI with Neo-Brutalist design system
- **Styling**: Custom theme with brutalist design principles
- **State Management**: React hooks with axios for API communication

## Key Features

### Hierarchical Semantic Chunking

- Multi-level document segmentation (Levels 1-4)
- Context-aware chunk relationships (parent-child hierarchies)
- Semantic boundary detection preserving narrative structure

### Adaptive Retrieval Strategies

1. **Adaptive Strategy**: Automatically selects optimal retrieval approach
2. **Overview Strategy**: High-level thematic questions
3. **Detail Strategy**: Specific factual inquiries
4. **Character Strategy**: Character analysis and relationships
5. **Structural Strategy**: Narrative structure and plot analysis

### Document Processing Pipeline

- HTML parsing and semantic segmentation
- Hierarchical metadata generation
- FAISS index creation with multi-level embeddings
- Context formatting with parent-child relationships

## API Endpoints

### System Management

- `GET /` - API root endpoint
- `GET /status` - System health and status
- `POST /initialize` - Initialize RAG pipeline with document
- `POST /reset` - Reset system state
- `GET /hierarchy` - Retrieve document hierarchy structure

### Question Answering

- `POST /ask` - Submit questions with configurable retrieval parameters
  - Parameters: `question`, `k_chunks`, `similarity_threshold`, `retrieval_strategy`

## Data Flow

1. **Document Ingestion**: HTML parsing → Semantic segmentation → Hierarchy creation
2. **Embedding Generation**: Chunk encoding → FAISS index creation
3. **Query Processing**: Question classification → Strategy selection → Hierarchical retrieval
4. **Response Generation**: Context assembly → Prompt construction → LLM generation

## System Requirements

### Backend Dependencies

```
fastapi>=0.104.1
pydantic>=2.5.0
google-generativeai>=0.3.0
faiss-cpu>=1.7.4
python-dotenv>=1.0.0
uvicorn>=0.24.0
```

### Frontend Dependencies

```
react>=18.2.0
typescript>=5.0.0
@mui/material>=5.14.0
axios>=1.6.0
react-markdown>=9.0.0
```

## Configuration

### Environment Variables

Create `.env` file in backend root:

```
GOOGLE_API_KEY=your_google_api_key_here
```

### File Structure

```
odyssey-rag/
├── backend/
│   ├── main.py                 # FastAPI application
│   ├── src/
│   │   └── odyssey_hierarchical_pipeline.py  # Core RAG pipeline
│   ├── data/
│   │   ├── raw/               # Source HTML documents
│   │   └── processed/         # Processed chunks and indices
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   └── App.tsx            # React application
│   ├── public/
│   └── package.json
└── docker-compose.yml
```

## Deployment Architecture

### Production Stack

- **Web Server**: Nginx as reverse proxy and load balancer
- **Application Server**: Uvicorn with Gunicorn workers
- **Containerization**: Docker with multi-stage builds
- **Orchestration**: Docker Compose for local development

### Docker Configuration

Multi-container setup with:

1. **Backend Service**: Python FastAPI application
2. **Frontend Service**: Nginx serving React build
3. **Data Volume**: Persistent storage for FAISS indices

### Nginx Configuration

- Reverse proxy to backend API
- Static file serving for frontend
- SSL/TLS termination
- CORS and security headers
- Rate limiting and caching

## Installation and Setup

### Local Development

1. **Backend Setup**:

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```

2. **Frontend Setup**:

```bash
cd frontend
npm install
npm run dev
```

### Docker Deployment

```bash
# Build and run containers
docker-compose up --build

# Production build
docker-compose -f docker-compose.prod.yml up --build
```

## Performance Characteristics

### Retrieval Performance

- **FAISS Index**: Approximate nearest neighbor search
- **Query Latency**: < 500ms for typical queries
- **Index Size**: ~50MB for Odyssey text
- **Concurrent Users**: Designed for 100+ simultaneous users

### Scalability Features

- Stateless backend design
- FAISS index loading in memory
- Async API endpoints
- Horizontal scaling support

## Testing Strategy

### Unit Tests

- Document parsing and chunking
- Embedding generation
- Retrieval accuracy
- API endpoint functionality

### Integration Tests

- End-to-end question answering
- Hierarchical retrieval validation
- System initialization workflows
- Error handling scenarios

### Load Testing

- Concurrent user simulation
- API response time monitoring
- Memory usage profiling
- FAISS search performance

## Monitoring and Logging

### Application Metrics

- Request/response timings
- Retrieval strategy effectiveness
- Token usage tracking
- Error rates and types

### System Monitoring

- Container health checks
- Resource utilization (CPU, memory)
- API endpoint availability
- FAISS index loading status

## Security Considerations

### API Security

- CORS configuration
- Request validation with Pydantic
- API key management
- Input sanitization

### Data Security

- Environment variable management
- Secure file handling
- Index file encryption (planned)
- Audit logging

## Future Enhancements

### Short-term Roadmap

1. Advanced caching layer for frequent queries
2. User session management
3. Query history and favorites
4. Export functionality for results

### Long-term Roadmap

1. Multi-document support
2. Custom embedding model fine-tuning
3. Advanced analytics dashboard
4. Plugin architecture for new retrieval strategies
5. Mobile application version
