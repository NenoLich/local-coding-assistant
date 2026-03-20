# Dashboard Testing Guide

This document provides a comprehensive guide to testing the LOCCA dashboard functionality.

## Overview

The dashboard testing suite covers all aspects of the ExecutionFrame dashboard implementation:

- **Unit Tests**: Individual component testing
- **Integration Tests**: Component interaction testing  
- **End-to-End Tests**: Complete workflow testing

## Test Structure

```
tests/
├── unit/dashboard/           # Unit tests for dashboard components
├── integration/dashboard/    # Integration tests for dashboard workflows
├── e2e/dashboard/          # End-to-end tests for complete scenarios
└── conftest_dashboard.py   # Shared test fixtures and helpers
```

## Test Categories

### 1. Unit Tests (`tests/unit/dashboard/`)

#### Models Testing (`test_models.py`)
- **Purpose**: Validate Pydantic model schemas and data validation
- **Coverage**: All dashboard models (RunSummary, FrameDetail, DashboardStats, etc.)
- **Key Tests**:
  - Model creation with valid/invalid data
  - Serialization/deserialization
  - Default values and optional fields
  - Type validation

#### EventCollector Testing (`test_event_collector.py`)
- **Purpose**: Test the core event aggregation and storage logic
- **Coverage**: EventCollector class functionality
- **Key Tests**:
  - Event ingestion and processing
  - Session and run lifecycle management
  - Data aggregation and statistics
  - Concurrent event handling
  - Memory management and limits

#### CLI Commands Testing (`test_cli_commands.py`)
- **Purpose**: Test dashboard CLI command functionality
- **Coverage**: All dashboard CLI commands (serve, stop)
- **Key Tests**:
  - Command argument parsing
  - Process management (start/stop)
  - Subprocess handling
  - Error scenarios

#### Routes Testing (`test_routes.py`)
- **Purpose**: Test FastAPI routes and endpoints
- **Coverage**: Main routes, API routes, WebSocket routes
- **Key Tests**:
  - HTTP endpoint responses
  - WebSocket connection handling
  - Request/response validation
  - Error handling

### 2. Integration Tests (`tests/integration/dashboard/`)

#### App Integration (`test_app_integration.py`)
- **Purpose**: Test dashboard application startup and serving
- **Coverage**: FastAPI app lifecycle and configuration
- **Key Tests**:
  - App creation and configuration
  - Middleware setup
  - Route registration
  - Performance under load
  - Error recovery

#### WebSocket Integration (`test_websocket_integration.py`)
- **Purpose**: Test real-time WebSocket functionality
- **Coverage**: Connection management and event broadcasting
- **Key Tests**:
  - Multiple concurrent connections
  - Real-time event broadcasting
  - Message handling
  - Connection failure recovery

### 3. End-to-End Tests (`tests/e2e/dashboard/`)

#### Workflow Testing (`test_dashboard_workflow.py`)
- **Purpose**: Test complete dashboard workflows
- **Coverage**: Full user scenarios and data flows
- **Key Tests**:
  - Complete session lifecycle
  - Multiple concurrent sessions
  - Error recovery scenarios
  - Data integrity
  - Performance under realistic load

## Test Fixtures and Helpers

### Shared Fixtures (`tests/conftest_dashboard.py`)

#### Core Fixtures
- `dashboard_client`: FastAPI test client
- `mock_event_collector`: Mocked EventCollector instance
- `sample_execution_events`: Sample ExecutionEvent objects
- `sample_run_summary`: Sample RunSummary model
- `sample_frame_detail`: Sample FrameDetail model

#### Helper Classes
- `DashboardTestHelpers`: Utility methods for test data creation
- `performance_timer`: Performance measurement utilities

#### Patch Fixtures
- `patch_event_collector`: Mock EventCollector injection
- `patch_connection_manager`: Mock WebSocket manager

## Running Tests

### Run All Dashboard Tests
```bash
# Run all dashboard tests
pytest tests/unit/dashboard/ tests/integration/dashboard/ tests/e2e/dashboard/ -v

# Run with coverage
pytest tests/unit/dashboard/ tests/integration/dashboard/ tests/e2e/dashboard/ --cov=local_coding_assistant.dashboard --cov-report=html
```

### Run Specific Test Categories
```bash
# Unit tests only
pytest tests/unit/dashboard/ -v

# Integration tests only
pytest tests/integration/dashboard/ -v

# E2E tests only
pytest tests/e2e/dashboard/ -v
```

### Run Specific Test Files
```bash
# Models testing
pytest tests/unit/dashboard/test_models.py -v

# EventCollector testing
pytest tests/unit/dashboard/test_event_collector.py -v

# CLI testing
pytest tests/unit/dashboard/test_cli_commands.py -v
```

### Run with Markers
```bash
# Run async tests only
pytest -m asyncio tests/unit/dashboard/ -v

# Run performance tests
pytest -m "performance" tests/ -v
```

## Test Data Management

### Sample Data Creation
Use the `DashboardTestHelpers` class to create consistent test data:

```python
def test_with_sample_data(dashboard_helpers):
    # Create complete session events
    events = dashboard_helpers.create_complete_session_events(
        session_id="test-session",
        run_id="test-run", 
        frame_id="test-frame"
    )
    
    # Create event data for API
    event_data = dashboard_helpers.create_event_data(
        event_type="session_start",
        session_id="test-session",
        data={"user_query": "test"}
    )
```

### Mock Configuration
Use the patch fixtures to mock dependencies:

```python
def test_with_mock_collector(patch_event_collector):
    # Configure mock return values
    patch_event_collector.get_dashboard_stats.return_value = {
        "total_runs": 10,
        "success_rate": "85%"
    }
    
    # Test implementation
    # ...
```

## Performance Testing

### Load Testing Patterns
```python
def test_high_volume_events(dashboard_client, performance_timer):
    performance_timer.start()
    
    # Send many events
    for i in range(1000):
        response = dashboard_client.post("/api/events/ingest", json=event_data)
        assert response.status_code == 200
    
    performance_timer.stop()
    assert performance_timer.duration < 10.0  # Should complete in < 10s
```

### Concurrent Testing
```python
def test_concurrent_sessions(dashboard_client):
    import threading
    
    def create_session(session_id):
        # Create session events
        pass
    
    threads = []
    for i in range(10):
        thread = threading.Thread(target=create_session, args=(f"session-{i}",))
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()
```

## WebSocket Testing

### Connection Testing
```python
def test_websocket_connection(dashboard_client):
    with dashboard_client.websocket_connect("/ws/ws") as websocket:
        # Receive initial data
        data = websocket.receive_json()
        assert data["type"] == "stats_update"
        
        # Send ping
        websocket.send_json({"type": "ping"})
        response = websocket.receive_json()
        assert response["type"] == "pong"
```

### Real-time Testing
```python
def test_real_time_broadcasting(dashboard_client, patch_connection_manager):
    # Setup mock connections
    mock_ws = dashboard_helpers.create_mock_websocket()
    patch_connection_manager.active_connections = [mock_ws]
    
    # Send event via API
    response = dashboard_client.post("/api/events/ingest", json=event_data)
    assert response.status_code == 200
    
    # Verify broadcast
    mock_ws.send_text.assert_called_once()
```

## Error Handling Testing

### API Error Scenarios
```python
def test_api_error_handling(dashboard_client, error_scenarios):
    # Test invalid timestamp
    response = dashboard_client.post("/api/events/ingest", 
                                   json=error_scenarios["invalid_timestamp"])
    # Should handle gracefully
    
    # Test missing required fields
    response = dashboard_client.post("/api/events/ingest",
                                   json=error_scenarios["missing_session_id"])
    # Should return appropriate error
```

### WebSocket Error Handling
```python
def test_websocket_error_recovery(dashboard_client, patch_connection_manager):
    # Simulate connection failure
    patch_connection_manager.connect.side_effect = Exception("Connection failed")
    
    with pytest.raises(Exception):
        with dashboard_client.websocket_connect("/ws/ws"):
            pass
```

## Data Integrity Testing

### End-to-End Data Flow
```python
def test_data_integrity_workflow(dashboard_client):
    # Send complete event sequence
    events = create_complete_session_events()
    
    for event in events:
        response = dashboard_client.post("/api/events/ingest", json=event)
        assert response.status_code == 200
    
    # Verify data consistency
    response = dashboard_client.get("/api/runs")
    runs_data = response.json()
    
    # Validate data integrity
    assert len(runs_data["items"]) == 1
    assert runs_data["items"][0]["events_count"] == len(events)
```

## Best Practices

### Test Organization
1. **Use descriptive test names** that explain what is being tested
2. **Group related tests** in test classes
3. **Use fixtures** for common setup/teardown
4. **Mock external dependencies** to isolate units under test

### Test Data Management
1. **Use helper functions** for consistent test data creation
2. **Avoid hardcoded values** in tests
3. **Clean up resources** in teardown
4. **Use realistic data** that matches production scenarios

### Performance Considerations
1. **Set reasonable timeouts** for async operations
2. **Limit concurrent operations** in test environments
3. **Mock expensive operations** (like database calls)
4. **Measure and assert performance** where critical

### Error Testing
1. **Test both happy path and error scenarios**
2. **Verify graceful error handling**
3. **Test edge cases and boundary conditions**
4. **Validate error messages and status codes**

## Coverage Goals

Target coverage areas:
- **Models**: 100% coverage of all Pydantic models
- **EventCollector**: 95%+ coverage of core logic
- **CLI Commands**: 90%+ coverage of command flows
- **API Routes**: 95%+ coverage of endpoints
- **WebSocket**: 90%+ coverage of real-time features
- **Integration**: 80%+ coverage of component interactions

## Continuous Integration

### CI Pipeline Integration
```yaml
# Example GitHub Actions step
- name: Run Dashboard Tests
  run: |
    pytest tests/unit/dashboard/ tests/integration/dashboard/ tests/e2e/dashboard/ \
      --cov=local_coding_assistant.dashboard \
      --cov-fail-under=80 \
      --junitxml=dashboard-test-results.xml
```

### Test Reporting
- Generate HTML coverage reports
- Create test result summaries
- Monitor test execution times
- Track flaky tests

## Troubleshooting

### Common Issues

1. **Async Test Failures**
   - Ensure `@pytest.mark.asyncio` marker is used
   - Check event loop setup in fixtures
   - Verify proper async/await usage

2. **Mock Configuration**
   - Ensure mocks are properly patched
   - Check mock call expectations
   - Verify mock return values

3. **WebSocket Test Issues**
   - Check WebSocket connection setup
   - Verify message format expectations
   - Ensure proper cleanup

4. **Performance Test Flakiness**
   - Adjust timeout values
   - Use consistent test data sizes
   - Account for system load variations

### Debugging Tips

1. **Use verbose output**: `pytest -v -s`
2. **Enable debugging**: `pytest --pdb`
3. **Check mock calls**: `mock_collector.call_args_list`
4. **Log test data**: Add print statements for debugging
5. **Use breakpoints**: `import pdb; pdb.set_trace()`

## Future Enhancements

### Planned Test Improvements
1. **Visual Regression Testing**: Compare dashboard UI outputs
2. **Load Testing**: Simulate high user traffic
3. **Security Testing**: Validate authentication and authorization
4. **Browser Testing**: End-to-end browser automation
5. **API Contract Testing**: Validate API specifications

### Test Infrastructure
1. **Test Data Factory**: Generate realistic test data
2. **Custom Assertions**: Dashboard-specific test helpers
3. **Test Utilities**: Reusable test patterns
4. **Performance Baselines**: Track performance over time
5. **Automated Test Data**: Self-cleaning test environments
