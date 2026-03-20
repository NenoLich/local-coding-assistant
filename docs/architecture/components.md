# Components

Detailed breakdown of LOCCA's core components and their responsibilities.

## Agent Components

### Frame Agent
- **Purpose**: Main agent implementation orchestrating task execution
- **Responsibilities**:
  - Request processing and routing
  - Tool coordination and execution
  - Response synthesis
- **Features**: Flexible execution frames, context awareness, tool integration

### LLM Service
- **Purpose**: Unified interface for language model interactions
- **Responsibilities**:
  - Request formatting and validation
  - Streaming response handling
  - Provider abstraction
- **Features**: Multi-provider support, fallback routing

## Provider System

### Provider Router
- **Purpose**: Intelligent routing between LLM providers
- **Responsibilities**:
  - Health monitoring and failover
  - Load balancing and optimization
  - Cost management
- **Strategies**: Round-robin, priority-based, cost-optimized

### Provider Adapters
- **Purpose**: Provider-specific implementations
- **Responsibilities**:
  - API protocol handling
  - Authentication management
  - Response normalization
- **Supported**: OpenAI, Anthropic, local models, custom providers

## Tool System

### Tool Registry
- **Purpose**: Dynamic tool discovery and management
- **Responsibilities**:
  - Tool loading and validation
  - Schema enforcement
  - Metadata management
- **Features**: Hot reloading, dependency resolution

### Sandbox Manager
- **Purpose**: Secure tool execution environment
- **Responsibilities**:
  - Container lifecycle management
  - Resource isolation and limits
  - Security policy enforcement
- **Technologies**: Docker, process isolation

## Runtime Management

### Session Manager
- **Purpose**: Context persistence across interactions
- **Responsibilities**:
  - Session lifecycle management
  - Context storage and retrieval
  - Memory optimization
- **Features**: Automatic cleanup, serialization

### Execution Engine
- **Purpose**: Asynchronous task orchestration
- **Responsibilities**:
  - Concurrent execution management
  - Resource allocation
  - Error propagation
- **Patterns**: Async/await, task groups, cancellation

## Configuration System

### Config Manager
- **Purpose**: Hierarchical configuration management
- **Responsibilities**:
  - Layer merging and precedence
  - Validation and type checking
  - Runtime overrides
- **Layers**: Global, session, call-level

### Path Manager
- **Purpose**: Cross-platform path resolution
- **Responsibilities**:
  - Alias expansion (@root, @config, etc.)
  - Path normalization
  - Environment awareness
- **Features**: OS-independent, configurable aliases

## Infrastructure Components

### Logging System
- **Purpose**: Structured application logging
- **Responsibilities**:
  - Log aggregation and formatting
  - Level management
  - Output routing
- **Integration**: Structlog, JSON formatting

### Error Handler
- **Purpose**: Centralized error processing
- **Responsibilities**:
  - Exception catching and classification
  - Recovery strategies
  - User-friendly error messages
- **Features**: Custom exceptions, safe entry points

### Statistics Manager
- **Purpose**: Performance monitoring and metrics
- **Responsibilities**:
  - Metric collection and aggregation
  - Performance analysis
  - Reporting generation
- **Data**: Execution times, resource usage, success rates

## Dashboard System

### Dashboard Application
- **Purpose**: Web-based observability interface
- **Responsibilities**:
  - Real-time monitoring of execution sessions
  - Historical data analysis and visualization
  - Interactive analytics and reporting
- **Technologies**: FastAPI, WebSocket, Jinja2 templates

### Event Collector
- **Purpose**: Execution event aggregation and storage
- **Responsibilities**:
  - Real-time event ingestion from RuntimeManager
  - In-memory data aggregation and indexing
  - WebSocket broadcasting for live updates
- **Features**: Configurable retention limits, session tracking

### Dashboard Integration
- **Purpose**: Bridge between RuntimeManager and Dashboard
- **Responsibilities**:
  - HTTP-based event forwarding to dashboard
  - Connection management and error handling
  - Data serialization for web consumption
- **Integration**: RuntimeManager.orchestrate() event hook

## CLI Components

### Command Parser
- **Purpose**: CLI argument processing
- **Responsibilities**:
  - Command routing and validation
  - Option parsing
  - Help generation
- **Framework**: Typer-based

### Rendering Engine
- **Purpose**: CLI output formatting
- **Responsibilities**:
  - Progress indicators
  - Table and list formatting
  - Color and styling
- **Library**: Rich

## Integration Points

### External APIs
- **Purpose**: Third-party service integration
- **Responsibilities**:
  - API client management
  - Authentication handling
  - Rate limiting
- **Examples**: GitHub, Docker, cloud providers

### Plugin System
- **Purpose**: Extensibility framework
- **Responsibilities**:
  - Plugin discovery and loading
  - Interface validation
  - Lifecycle management
- **Types**: Tools, providers, agents

## Next Steps

- [Data Flow](data-flow.md)
- [Contracts](contracts.md)
