# Dashboard API

The LOCCA Dashboard provides a comprehensive REST API for accessing execution data, analytics, and real-time updates. This document describes all available endpoints and their usage.

## Base URL

```
http://127.0.0.1:8080/api
```

## Authentication

Currently, the dashboard API does not require authentication (development mode). In production deployments, authentication middleware can be added.

## Common Response Formats

### Success Response
```json
{
  "status": "ok",
  "data": { ... }
}
```

### Error Response
```json
{
  "detail": "Error description",
  "status_code": 400
}
```

### Paginated Response
```json
{
  "items": [ ... ],
  "total": 150,
  "offset": 0,
  "limit": 100,
  "has_next": true,
  "has_prev": false
}
```

## Endpoints

### Events

#### Ingest Event
```http
POST /api/events/ingest
```

Receives execution events from agent processes.

**Request Body:**
```json
{
  "type": "frame_start",
  "session_id": "session_123",
  "frame_id": "frame_456",
  "data": {
    "prompt": "User query",
    "model": "gpt-4"
  },
  "timestamp": "2024-01-01T12:00:00Z"
}
```

**Response:**
```json
{
  "status": "ok",
  "event_type": "frame_start"
}
```

### System Status

#### API Status
```http
GET /api/status
```

Check API health and version.

**Response:**
```json
{
  "status": "ok",
  "version": "0.1.0"
}
```

#### Health Check
```http
GET /health
```

Application health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "service": "locca-dashboard"
}
```

### Statistics

#### Dashboard Stats
```http
GET /api/stats
```

Get overall dashboard statistics.

**Response:**
```json
{
  "total_runs": 42,
  "success_rate": "85.7%",
  "avg_duration": "12.5s",
  "active_sessions": 2,
  "completed_runs": 36,
  "error_runs": 6
}
```

#### Recent Activity
```http
GET /api/recent-activity?limit=50
```

Get recent activity feed.

**Query Parameters:**
- `limit` (int, optional): Number of items to return (max 200, default 50)

**Response:**
```json
{
  "activities": [
    {
      "run_id": "run_123",
      "status": "completed",
      "timestamp": "2024-01-01T12:00:00Z",
      "duration": 15.2
    }
  ],
  "last_updated": "2024-01-01T12:00:00Z"
}
```

### Runs

#### List Runs
```http
GET /api/runs?limit=100&offset=0&status=completed&date_from=2024-01-01&date_to=2024-01-02
```

List execution runs with filtering and pagination.

**Query Parameters:**
- `limit` (int, optional): Items per page (max 1000, default 100)
- `offset` (int, optional): Pagination offset (default 0)
- `status` (string, optional): Filter by status (running, completed, error)
- `date_from` (datetime, optional): Filter from date (ISO 8601)
- `date_to` (datetime, optional): Filter to date (ISO 8601)
- `model` (string, optional): Filter by model name
- `provider` (string, optional): Filter by provider name

**Response:**
```json
{
  "items": [
    {
      "run_id": "run_123",
      "session_id": "session_456",
      "status": "completed",
      "start_time": "2024-01-01T12:00:00Z",
      "end_time": "2024-01-01T12:00:15Z",
      "duration": 15.2,
      "events_count": 25,
      "tokens_used": 1250
    }
  ],
  "total": 42,
  "offset": 0,
  "limit": 100,
  "has_next": false,
  "has_prev": false
}
```

#### Get Run Details
```http
GET /api/runs/{run_id}
```

Get detailed information for a specific run.

**Path Parameters:**
- `run_id` (string): Unique identifier for the run

**Response:**
```json
{
  "run_id": "run_123",
  "session_id": "session_456",
  "status": "completed",
  "start_time": "2024-01-01T12:00:00Z",
  "end_time": "2024-01-01T12:00:15Z",
  "duration": 15.2,
  "events_count": 25,
  "tokens_used": 1250,
  "events": [
    {
      "type": "frame_start",
      "timestamp": "2024-01-01T12:00:00Z",
      "data": { ... }
    }
  ],
  "final_answer": "The answer to the user query",
  "error_message": null
}
```

#### Export Runs
```http
GET /api/runs/export?format=csv&status=completed&date_from=2024-01-01
```

Export runs data in various formats.

**Query Parameters:**
- `format` (string): Export format (csv, json)
- `status` (string, optional): Filter by status
- `date_from` (datetime, optional): Filter from date
- `date_to` (datetime, optional): Filter to date
- `model` (string, optional): Filter by model
- `provider` (string, optional): Filter by provider

**Response (CSV):**
```csv
run_id,session_id,status,start_time,end_time,duration,tokens_used
run_123,session_456,completed,2024-01-01T12:00:00Z,2024-01-01T12:00:15Z,15.2,1250
```

**Response (JSON):**
```json
[
  {
    "run_id": "run_123",
    "session_id": "session_456",
    "status": "completed",
    "start_time": "2024-01-01T12:00:00Z",
    "end_time": "2024-01-01T12:00:15Z",
    "duration": 15.2,
    "tokens_used": 1250
  }
]
```

### Frames

#### List Frames
```http
GET /api/frames?limit=100&offset=0&status=completed&run_id=run_123
```

List frames with filtering and pagination.

**Query Parameters:**
- `limit` (int, optional): Items per page (max 200, default 100)
- `offset` (int, optional): Pagination offset (default 0)
- `status` (string, optional): Filter by status
- `run_id` (string, optional): Filter by run ID
- `date_from` (datetime, optional): Filter from date
- `date_to` (datetime, optional): Filter to date

**Response:**
```json
{
  "items": [
    {
      "frame_id": "frame_789",
      "run_id": "run_123",
      "status": "completed",
      "start_time": "2024-01-01T12:00:00Z",
      "end_time": "2024-01-01T12:00:05Z",
      "duration": 5.1,
      "action_count": 3
    }
  ],
  "total": 25,
  "offset": 0,
  "limit": 100,
  "has_next": false,
  "has_prev": false
}
```

#### Get Frame Details
```http
GET /api/frames/{frame_id}
```

Get detailed information for a specific frame.

**Path Parameters:**
- `frame_id` (string): Unique identifier for the frame

**Response:**
```json
{
  "frame_id": "frame_789",
  "run_id": "run_123",
  "status": "completed",
  "start_time": "2024-01-01T12:00:00Z",
  "end_time": "2024-01-01T12:00:05Z",
  "duration": 5.1,
  "action_count": 3,
  "prompt_context": "User query context",
  "llm_response": "LLM generated response",
  "tool_calls": [
    {
      "tool": "calculator",
      "input": { "expression": "2+2" },
      "output": "4",
      "timestamp": "2024-01-01T12:00:02Z"
    }
  ],
  "actions": [
    {
      "type": "tool_call",
      "tool": "calculator",
      "timestamp": "2024-01-01T12:00:02Z",
      "duration": 0.1
    }
  ]
}
```

#### Export Frames
```http
GET /api/frames/export?format=csv&run_id=run_123
```

Export frames data in various formats.

**Query Parameters:**
- `format` (string): Export format (csv, json)
- `run_id` (string, optional): Filter by run ID
- `status` (string, optional): Filter by status
- `date_from` (datetime, optional): Filter from date
- `date_to` (datetime, optional): Filter to date

### Analytics

#### Get Analytics Data
```http
GET /api/analytics?days=30&model=gpt-4
```

Get analytical data and metrics.

**Query Parameters:**
- `days` (int, optional): Number of days to analyze (max 365, default 30)
- `model` (string, optional): Filter by model
- `provider` (string, optional): Filter by provider

**Response:**
```json
{
  "success_rate_trend": [
    { "date": "2024-01-01", "rate": 0.85 },
    { "date": "2024-01-02", "rate": 0.87 }
  ],
  "duration_trend": [
    { "date": "2024-01-01", "avg_duration": 12.5 },
    { "date": "2024-01-02", "avg_duration": 11.8 }
  ],
  "token_usage": {
    "total": 50000,
    "by_model": {
      "gpt-4": 30000,
      "gpt-3.5-turbo": 20000
    }
  },
  "top_errors": [
    { "error": "Timeout error", "count": 5 },
    { "error": "API limit exceeded", "count": 3 }
  ]
}
```

## WebSocket API

### Real-time Events

#### Event Stream
```javascript
const ws = new WebSocket('ws://127.0.0.1:8080/ws/events');

ws.onmessage = function(event) {
  const data = JSON.parse(event.data);
  console.log('New event:', data);
};
```

**Message Format:**
```json
{
  "type": "frame_start",
  "session_id": "session_123",
  "frame_id": "frame_456",
  "data": { ... },
  "timestamp": "2024-01-01T12:00:00Z"
}
```

#### Live Sessions
```javascript
const ws = new WebSocket('ws://127.0.0.1:8080/ws/sessions');

ws.onmessage = function(event) {
  const data = JSON.parse(event.data);
  console.log('Session update:', data);
};
```

**Message Format:**
```json
{
  "session_id": "session_123",
  "status": "running",
  "active_frames": 2,
  "total_events": 15,
  "last_activity": "2024-01-01T12:00:00Z"
}
```

## Error Codes

| Status Code | Description |
|-------------|-------------|
| 200 | Success |
| 400 | Bad Request |
| 404 | Not Found |
| 422 | Validation Error |
| 500 | Internal Server Error |

## Rate Limiting

Currently, no rate limiting is implemented in development mode. For production deployments, consider implementing rate limiting middleware.

## SDK Examples

### Python

```python
import httpx
import websocket

class DashboardClient:
    def __init__(self, base_url="http://127.0.0.1:8080"):
        self.base_url = base_url
        self.client = httpx.Client()
    
    def get_runs(self, status=None, limit=100):
        params = {"limit": limit}
        if status:
            params["status"] = status
        
        response = self.client.get(f"{self.base_url}/api/runs", params=params)
        response.raise_for_status()
        return response.json()
    
    def get_run_details(self, run_id):
        response = self.client.get(f"{self.base_url}/api/runs/{run_id}")
        response.raise_for_status()
        return response.json()

# Usage
client = DashboardClient()
runs = client.get_runs(status="completed")
print(f"Found {runs['total']} completed runs")
```

### JavaScript

```javascript
class DashboardClient {
  constructor(baseUrl = 'http://127.0.0.1:8080') {
    this.baseUrl = baseUrl;
  }
  
  async getRuns(status = null, limit = 100) {
    const params = new URLSearchParams({ limit: limit.toString() });
    if (status) params.append('status', status);
    
    const response = await fetch(`${this.baseUrl}/api/runs?${params}`);
    return await response.json();
  }
  
  async getRunDetails(runId) {
    const response = await fetch(`${this.baseUrl}/api/runs/${runId}`);
    return await response.json();
  }
}

// Usage
const client = new DashboardClient();
const runs = await client.getRuns('completed');
console.log(`Found ${runs.total} completed runs`);
```

## Integration with Runtime

### Enabling Dashboard Integration

```python
from local_coding_assistant.runtime.dashboard_integration import enable_dashboard_integration

# Enable dashboard integration
enable_dashboard_integration(dashboard_url="http://127.0.0.1:8080")

# Events will now be automatically sent to the dashboard
```

### Custom Event Collection

```python
from local_coding_assistant.runtime.dashboard_integration import collect_event_for_dashboard
from local_coding_assistant.runtime.events import ExecutionEvent, EventType

# Create and send custom event
event = ExecutionEvent(
    type=EventType.CUSTOM,
    session_id="session_123",
    frame_id="frame_456",
    data={"message": "Custom event data"},
    timestamp=datetime.now()
)

await collect_event_for_dashboard(event)
```
