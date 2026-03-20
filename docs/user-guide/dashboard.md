# Dashboard

The LOCCA Dashboard provides a web-based interface for monitoring and analyzing agent execution sessions. It offers real-time observability, historical analysis, and performance insights for the ExecutionFrame system.

## Features

### Real-time Monitoring
- **Live Sessions**: Monitor active execution sessions as they run
- **WebSocket Updates**: Real-time updates without page refresh
- **Progress Tracking**: Visual progress indicators for ongoing operations
- **Event Streaming**: Live stream of execution events

### Historical Analysis
- **Runs List**: Browse and filter historical execution runs
- **Run Details**: Deep dive into individual runs with complete event timelines
- **Frame Analysis**: Examine individual frames with prompts, responses, and tool calls
- **Performance Metrics**: Duration, token usage, and success rates

### Analytics & Insights
- **Dashboard Overview**: Key metrics and trends at a glance
- **Interactive Charts**: Visualizations for performance trends and patterns
- **Filtering & Search**: Advanced filtering by date, status, model, and more
- **Data Export**: Export data in CSV or JSON formats

## Quick Start

### Starting the Dashboard

```bash
# Start dashboard on default port 8080
uv run locca dashboard serve

# Start on custom host/port
uv run locca dashboard serve --host 0.0.0.0 --port 9000

# Start in background (detached mode)
uv run locca dashboard serve --detach

# Start with auto-reload for development
uv run locca dashboard serve --reload
```

### Stopping the Dashboard

```bash
# Stop dashboard on specific port
uv run locca dashboard stop --port 8080

# Stop all dashboard servers
uv run locca dashboard stop --all
```

### Accessing the Dashboard

Once started, access the dashboard at:
- `http://127.0.0.1:8080` (default)
- `http://0.0.0.0:8080` (if using `--host 0.0.0.0`)

## Dashboard Pages

### Homepage (`/`)
- Overview of recent activity
- Quick access to main features
- System health indicators

### Runs List (`/runs`)
- Paginated list of all execution runs
- Filtering by status, date range, model, provider
- Sortable columns (duration, tokens, timestamp)
- Quick actions for run details

### Run Detail (`/runs/{run_id}`)
- Complete timeline of events in the run
- Performance metrics and statistics
- Error information (if applicable)
- Links to individual frames

### Frame Detail (`/frames/{frame_id}`)
- Prompt context and LLM response
- Tool calls made during the frame
- Action timeline with timestamps
- Resource usage metrics

### Analytics (`/analytics`)
- Success rates and trends
- Performance charts and graphs
- Token usage analysis
- Model and provider comparisons

### Live Monitoring (`/live`)
- Real-time view of active sessions
- Live progress updates
- Current frame status
- Session metrics

## Architecture

### Components

#### FastAPI Application
- **app.py**: Main FastAPI application configuration
- **routes/**: API and web route handlers
- **templates/**: Jinja2 HTML templates
- **static/**: Static assets (CSS, JS, images)

#### Data Layer
- **event_collector.py**: In-memory event aggregation and storage
- **models.py**: Pydantic models for data structures
- **Runtime Integration**: Event collection from RuntimeManager

#### Real-time Features
- **WebSocket Handler**: Real-time event broadcasting
- **Live Updates**: Automatic page updates without refresh
- **Session Monitoring**: Active session tracking

### Data Flow

1. **Event Generation**: RuntimeManager generates ExecutionEvents
2. **Event Collection**: DashboardIntegration forwards events to dashboard
3. **Event Processing**: EventCollector aggregates and stores events
4. **Real-time Broadcasting**: WebSocket updates connected clients
5. **API Responses**: REST endpoints provide data for UI components

## Configuration

### Environment Variables

```bash
# Dashboard configuration
LOCCA_DASHBOARD_HOST=127.0.0.1
LOCCA_DASHBOARD_PORT=8080
LOCCA_DASHBOARD_LOG_LEVEL=INFO
```

### Dashboard Settings

The dashboard can be configured through:

- **CLI Flags**: Override settings per command
- **Environment Variables**: Persistent configuration
- **Default Values**: Sensible defaults for development

## API Reference

### REST Endpoints

#### Events
- `POST /api/events/ingest`: Ingest events from agent processes
- `GET /api/events`: List events with filtering

#### Runs
- `GET /api/runs`: List runs with pagination and filtering
- `GET /api/runs/{run_id}`: Get detailed run information

#### Frames
- `GET /api/frames`: List frames with filtering
- `GET /api/frames/{frame_id}`: Get detailed frame information

#### Analytics
- `GET /api/stats`: Dashboard statistics
- `GET /api/recent-activity`: Recent activity feed
- `GET /api/analytics`: Analytical data and metrics

#### System
- `GET /api/status`: API health check
- `GET /health`: Application health check

### WebSocket Endpoints

#### Real-time Updates
- `WS /ws/events`: Real-time event streaming
- `WS /ws/sessions`: Live session updates

## Development

### Running in Development Mode

```bash
# Start with auto-reload
uv run locca dashboard serve --reload --log-level DEBUG

# Start with custom port for development
uv run locca dashboard serve --port 8081 --reload
```

### Adding New Features

1. **Routes**: Add new endpoints in `routes/api.py` or `routes/main.py`
2. **Models**: Define data structures in `models.py`
3. **Templates**: Create HTML templates in `templates/`
4. **Event Processing**: Extend `event_collector.py` for new data types

### Testing

```bash
# Run dashboard-specific tests
uv run pytest tests/unit/dashboard/
uv run pytest tests/integration/dashboard/

# Run all tests
uv run pytest
```

## Troubleshooting

### Common Issues

#### Dashboard Won't Start
- Check if port is already in use: `uv run locca dashboard stop --all`
- Verify dependencies: `uv pip install -e ".[dev]"`
- Check logs: Use `--log-level DEBUG` for detailed output

#### Events Not Appearing
- Verify dashboard integration is enabled in agent
- Check network connectivity to dashboard URL
- Review event collector logs for processing errors

#### Performance Issues
- Reduce event retention limits in EventCollector
- Enable pagination for large datasets
- Consider database persistence for production use

### Debug Mode

```bash
# Start with debug logging
uv run locca dashboard serve --log-level DEBUG

# Check dashboard integration status
curl http://127.0.0.1:8080/api/status
```

## Security Considerations

### Development Mode
- No authentication required
- CORS enabled for all origins
- Suitable for local development only

### Production Deployment
- Implement authentication middleware
- Restrict CORS origins
- Use HTTPS for secure connections
- Consider rate limiting for API endpoints

## Future Enhancements

### Planned Features
- **Database Persistence**: SQLite/PostgreSQL backend for historical data
- **User Authentication**: Multi-user support with permissions
- **Advanced Analytics**: ML-based insights and predictions
- **Alerting**: Configurable alerts for failures and performance issues
- **Export Formats**: Additional export formats (PDF, Excel)
- **API Tokens**: Secure API access for external integrations

### Integration Opportunities
- **External Monitoring**: Grafana, Prometheus integration
- **CI/CD Pipelines**: Automated testing and deployment insights
- **Collaboration**: Multi-user session sharing and commenting
