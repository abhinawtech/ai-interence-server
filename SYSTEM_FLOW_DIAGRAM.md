# AI Inference Server - Detailed System Flow Diagram

## 🌊 Complete Request Processing Flow

```mermaid
graph TD
    %% Client Request Entry
    A[Client HTTP Request] -->|POST /api/v1/generate| B[Axum Web Server]
    
    %% Request Validation & Processing
    B --> C{Request Validation}
    C -->|Invalid| D[Return 400 Error]
    C -->|Valid| E[Generate Request ID]
    
    %% Batch Processing Decision
    E --> F[Submit to BatchProcessor]
    F --> G{Queue Status Check}
    G -->|Queue Full| H[Return 503 Service Unavailable]
    G -->|Queue Available| I[Add to Request Queue]
    
    %% Batch Collection Strategy
    I --> J[Batch Collection Loop]
    J --> K{Batch Conditions}
    K -->|Max Size Reached<br/>OR Max Wait Time| L[Process Batch]
    K -->|Continue Waiting| M[Wait 5ms]
    M --> K
    
    %% Model Processing Path
    L --> N{Batch Size = 1?}
    N -->|Yes| O[Single Request Fast Path]
    N -->|No| P[Multi-Request Batch Path]
    
    %% Single Request Processing
    O --> Q[Acquire Model Lock]
    Q --> R[Model Generate Call]
    R --> S[Release Model Lock Early]
    S --> T[Send Response]
    
    %% Batch Request Processing
    P --> U[Acquire Model Lock]
    U --> V[Process All Requests Sequentially]
    V --> W[Release Model Lock]
    W --> X[Send All Responses]
    
    %% Model Generation Deep Dive
    R --> AA[Format Chat Prompt]
    V --> AA
    AA --> BB[Tokenize Input]
    BB --> CC[Create Input Tensor]
    CC --> DD[Initialize KV Cache]
    DD --> EE[First Forward Pass - Full Prompt]
    
    %% Token Generation Loop
    EE --> FF[Extract First Token]
    FF --> GG{Generation Loop}
    GG -->|Continue| HH[Create Single Token Tensor]
    HH --> II[Forward Pass with Cache]
    II --> JJ[Sample Next Token]
    JJ --> KK{EOS Token OR Max Tokens?}
    KK -->|No| GG
    KK -->|Yes| LL[Decode Tokens to Text]
    
    %% Performance Monitoring
    LL --> MM[Calculate Performance Metrics]
    MM --> NN[Log Performance Stats]
    NN --> T
    NN --> X
    
    %% Response Construction
    T --> OO[Build JSON Response]
    X --> OO
    OO --> PP[Return HTTP 200]
    
    %% Error Handling Paths
    Q -->|Lock Failed| QQ[Handle Lock Error]
    U -->|Lock Failed| QQ
    AA -->|Tokenization Failed| RR[Handle Token Error]
    EE -->|Model Error| SS[Handle Model Error]
    II -->|Forward Pass Error| SS
    
    QQ --> TT[Return 500 Error]
    RR --> TT
    SS --> TT
    
    %% Background Processes
    subgraph "Background Tasks"
        YY[Batch Processing Loop]
        ZZ[Statistics Updates]
        AAA[Health Check Monitoring]
    end
    
    %% Styling
    classDef clientRequest fill:#e1f5fe
    classDef validation fill:#f3e5f5
    classDef batchProcess fill:#e8f5e8
    classDef modelProcess fill:#fff3e0
    classDef errorHandle fill:#ffebee
    classDef response fill:#f1f8e9
    
    class A clientRequest
    class C,E validation
    class F,G,I,J,K,L,N batchProcess
    class Q,R,AA,BB,CC,DD,EE,FF,GG,HH,II,JJ,KK,LL,MM modelProcess
    class D,H,QQ,RR,SS,TT errorHandle
    class OO,PP,T,X response
```

## 🔄 Detailed Component Interactions

```mermaid
sequenceDiagram
    participant C as Client
    participant W as Axum Web Server
    participant BP as BatchProcessor
    participant Q as Request Queue
    participant M as TinyLlama Model
    participant GPU as Metal GPU
    
    %% Request Flow
    C->>W: POST /api/v1/generate
    W->>W: Validate Request
    W->>BP: Submit Request
    BP->>Q: Add to Queue
    
    %% Batch Processing
    Note over BP,Q: Batch Collection Phase
    BP->>Q: Collect Requests
    BP->>BP: Check Batch Conditions
    
    %% Model Processing
    BP->>M: Acquire Lock
    Note over M,GPU: Model Inference Phase
    M->>M: Format Prompt
    M->>M: Tokenize Input
    M->>GPU: Create Tensors on Metal
    M->>GPU: Initialize KV Cache
    
    %% Generation Loop
    loop Token Generation
        M->>GPU: Forward Pass
        GPU->>M: Logits
        M->>M: Sample Token
        M->>M: Check EOS/Max Tokens
    end
    
    M->>M: Decode Tokens
    M->>M: Calculate Metrics
    M->>BP: Return Result
    BP->>W: Send Response
    W->>C: HTTP 200 + JSON
    
    %% Error Scenarios
    alt Validation Error
        W->>C: HTTP 400 Bad Request
    else Queue Full
        BP->>W: Service Unavailable
        W->>C: HTTP 503
    else Model Error
        M->>BP: Error Result
        BP->>W: Internal Error
        W->>C: HTTP 500
    end
```

## ⚡ Performance Optimization Flow

```mermaid
graph LR
    %% Memory Optimization
    subgraph "Memory Optimizations"
        A1[SafeTensors Memory Mapping] --> A2[F16 Precision]
        A2 --> A3[Pre-allocated Vectors]
        A3 --> A4[Efficient Tensor Reuse]
    end
    
    %% GPU Acceleration
    subgraph "Metal GPU Acceleration"
        B1[Metal Device Selection] --> B2[Unified Memory Architecture]
        B2 --> B3[GPU Tensor Operations]
        B3 --> B4[Optimized Matrix Multiplication]
    end
    
    %% Batching Strategy
    subgraph "Batching Optimizations"
        C1[Request Aggregation] --> C2[Shared Model Lock]
        C2 --> C3[Single Request Fast Path]
        C3 --> C4[Early Lock Release]
    end
    
    %% Cache Strategy
    subgraph "KV Cache Optimization"
        D1[Cache Initialization] --> D2[Sequential Token Generation]
        D2 --> D3[Cache State Reuse]
        D3 --> D4[Memory Efficient Storage]
    end
    
    A4 --> E[13.8 tok/s Peak Performance]
    B4 --> E
    C4 --> E
    D4 --> E
```

## 🔧 System Architecture Overview

```mermaid
graph TB
    %% External Layer
    subgraph "External Interface"
        HTTP[HTTP API Endpoints]
        Health[Health Check]
        Batch[Batch Status]
    end
    
    %% Application Layer
    subgraph "Application Layer"
        Router[Axum Router]
        Middleware[Request Middleware]
        Validation[Input Validation]
    end
    
    %% Business Logic Layer
    subgraph "Business Logic"
        BatchProc[Batch Processor]
        Queue[Request Queue]
        Stats[Performance Statistics]
    end
    
    %% Model Layer
    subgraph "AI Model Layer"
        ModelMgr[Model Manager]
        Tokenizer[HuggingFace Tokenizer]
        LlamaModel[TinyLlama Model]
    end
    
    %% Infrastructure Layer
    subgraph "Infrastructure"
        Metal[Metal GPU]
        Memory[Unified Memory]
        Cache[KV Cache]
    end
    
    %% Data Flow
    HTTP --> Router
    Health --> Router
    Batch --> Router
    
    Router --> Middleware
    Middleware --> Validation
    Validation --> BatchProc
    
    BatchProc --> Queue
    BatchProc --> Stats
    Queue --> ModelMgr
    
    ModelMgr --> Tokenizer
    ModelMgr --> LlamaModel
    LlamaModel --> Metal
    
    Metal --> Memory
    Metal --> Cache
```

## 📊 Performance Metrics Flow

```mermaid
graph TD
    %% Timing Measurement Points
    A[Request Start] --> B[Validation Time]
    B --> C[Queue Wait Time]
    C --> D[Model Lock Acquisition]
    D --> E[Tokenization Time]
    E --> F[Generation Start]
    
    %% Generation Metrics
    F --> G[First Token Latency]
    G --> H[Token Generation Loop]
    H --> I[Total Generation Time]
    I --> J[Decoding Time]
    
    %% Performance Calculations
    J --> K[Calculate tok/s]
    K --> L[Calculate Queue Efficiency]
    L --> M[Update Rolling Averages]
    
    %% Monitoring Output
    M --> N[Structured Logs]
    M --> O[Performance Response Fields]
    M --> P[Batch Statistics]
    
    %% Key Metrics
    subgraph "Key Performance Indicators"
        K1[12-14 tok/s Average]
        K2[<100ms Queue Time]
        K3[2-4 Avg Batch Size]
        K4[>95% Success Rate]
    end
    
    N --> K1
    O --> K2
    P --> K3
    P --> K4
```

## 🚦 Error Handling Flow

```mermaid
graph TD
    %% Error Sources
    A[Input Validation Error] --> G[Error Handler]
    B[Queue Full Error] --> G
    C[Model Loading Error] --> G
    D[Generation Error] --> G
    E[GPU/Metal Error] --> G
    F[Timeout Error] --> G
    
    %% Error Processing
    G --> H{Error Type Classification}
    
    %% Error Response Mapping
    H -->|4xx Client Error| I[Bad Request Response]
    H -->|5xx Server Error| J[Internal Server Error]
    H -->|503 Service Error| K[Service Unavailable]
    
    %% Response Generation
    I --> L[JSON Error Response]
    J --> L
    K --> L
    
    %% Logging & Monitoring
    L --> M[Structured Error Logging]
    M --> N[Error Rate Monitoring]
    N --> O[Alert Thresholds]
    
    %% Recovery Actions
    G --> P{Recoverable?}
    P -->|Yes| Q[Retry Logic]
    P -->|No| R[Graceful Degradation]
    
    Q --> S[Exponential Backoff]
    R --> T[Circuit Breaker Pattern]
```

---

## 📋 Flow Diagram Legend

- **🔵 Blue Nodes**: Client interactions and external interfaces
- **🟢 Green Nodes**: Successful processing paths
- **🟡 Yellow Nodes**: Decision points and conditionals  
- **🔴 Red Nodes**: Error conditions and handling
- **⚪ Gray Nodes**: Background processes and monitoring

## 🎯 Key Optimization Points

1. **Batch Collection Strategy**: Dynamic batching based on load
2. **Metal GPU Utilization**: Unified memory architecture benefits
3. **KV Cache Management**: Efficient attention state reuse
4. **Memory Allocation**: Pre-allocation and tensor reuse patterns
5. **Lock Management**: Early release for better concurrency