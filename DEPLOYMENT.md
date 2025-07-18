# GoQuant Sentiment Trader - Deployment Guide

This guide covers deployment options for the GoQuant Sentiment Trader system.

## Prerequisites

- Python 3.9 or higher
- Docker and Docker Compose (for containerized deployment)
- API keys for data sources (Twitter, Reddit, NewsAPI, etc.)

## Local Development Setup

### 1. Clone and Install

```bash
git clone <repository-url>
cd GoQuant
python install.py
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys
```

### 3. Initialize Database

```bash
source venv/bin/activate  # On Windows: venv\Scripts\activate
python -m goquant.database.init_db
```

### 4. Run the System

```bash
# Full system
goquant-sentiment run

# Individual components
goquant-sentiment collect --once
goquant-sentiment analyze
goquant-sentiment signals
goquant-sentiment dashboard
```

## Docker Deployment

### 1. Build and Run with Docker Compose

```bash
# Copy environment file
cp .env.example .env
# Edit .env with your configuration

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f goquant

# Stop services
docker-compose down
```

### 2. Services Included

- **PostgreSQL**: Primary database
- **Redis**: Caching and session storage
- **GoQuant App**: Main application
- **Nginx**: Reverse proxy and load balancer

### 3. Accessing the Application

- Dashboard: http://localhost (or your domain)
- Health Check: http://localhost/health
- Database: localhost:5432
- Redis: localhost:6379

## Production Deployment

### 1. Environment Configuration

```bash
# Production environment variables
ENVIRONMENT=production
LOG_LEVEL=INFO
DATABASE_URL=postgresql://user:password@host:5432/goquant
REDIS_URL=redis://host:6379/0

# API Keys (required)
TWITTER_BEARER_TOKEN=your_token
REDDIT_CLIENT_ID=your_id
REDDIT_CLIENT_SECRET=your_secret
NEWS_API_KEY=your_key
ALPHA_VANTAGE_API_KEY=your_key
```

### 2. SSL Configuration

1. Obtain SSL certificates
2. Place certificates in `./ssl/` directory
3. Uncomment HTTPS configuration in `nginx.conf`
4. Update `docker-compose.yml` to expose port 443

### 3. Scaling

```bash
# Scale the application
docker-compose up -d --scale goquant=3

# Use external database
# Update DATABASE_URL to point to external PostgreSQL instance
```

### 4. Monitoring

```bash
# View system health
curl http://localhost/health

# Monitor logs
docker-compose logs -f --tail=100 goquant

# Monitor resources
docker stats
```

## Cloud Deployment

### AWS Deployment

1. **ECS with Fargate**
   ```bash
   # Build and push to ECR
   aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account>.dkr.ecr.us-east-1.amazonaws.com
   docker build -t goquant .
   docker tag goquant:latest <account>.dkr.ecr.us-east-1.amazonaws.com/goquant:latest
   docker push <account>.dkr.ecr.us-east-1.amazonaws.com/goquant:latest
   ```

2. **RDS for Database**
   - Create PostgreSQL RDS instance
   - Update DATABASE_URL in environment

3. **ElastiCache for Redis**
   - Create Redis ElastiCache cluster
   - Update REDIS_URL in environment

### Google Cloud Platform

1. **Cloud Run**
   ```bash
   # Build and deploy
   gcloud builds submit --tag gcr.io/PROJECT_ID/goquant
   gcloud run deploy --image gcr.io/PROJECT_ID/goquant --platform managed
   ```

2. **Cloud SQL**
   - Create PostgreSQL Cloud SQL instance
   - Configure connection

### Azure Deployment

1. **Container Instances**
   ```bash
   # Deploy to Azure Container Instances
   az container create --resource-group myResourceGroup --name goquant --image goquant:latest
   ```

## Performance Optimization

### 1. Database Optimization

```sql
-- Create indexes for better performance
CREATE INDEX idx_sentiment_symbol_time ON sentiment_data(symbol, processed_at);
CREATE INDEX idx_signals_active ON trading_signals(is_active, generated_at);
CREATE INDEX idx_raw_data_source_time ON raw_data(source, collected_at);
```

### 2. Application Optimization

```python
# Increase batch sizes for better throughput
SENTIMENT_ANALYSIS_BATCH_SIZE=200
DATA_COLLECTION_INTERVAL=30

# Use connection pooling
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=30
```

### 3. Caching Strategy

```python
# Enable Redis caching
REDIS_URL=redis://localhost:6379/0

# Cache sentiment results
CACHE_SENTIMENT_RESULTS=true
CACHE_TTL_SECONDS=300
```

## Monitoring and Alerting

### 1. Health Checks

```bash
# Application health
curl http://localhost/health

# Database health
pg_isready -h localhost -p 5432

# Redis health
redis-cli ping
```

### 2. Metrics Collection

```python
# Enable metrics collection
ENABLE_METRICS=true
METRICS_PORT=9090

# Prometheus integration
PROMETHEUS_ENABLED=true
```

### 3. Log Management

```bash
# Centralized logging
LOG_FORMAT=json
LOG_DESTINATION=stdout

# Log rotation
LOG_MAX_SIZE=100MB
LOG_BACKUP_COUNT=5
```

## Backup and Recovery

### 1. Database Backup

```bash
# Automated backup
pg_dump -h localhost -U goquant_user goquant > backup_$(date +%Y%m%d).sql

# Restore
psql -h localhost -U goquant_user goquant < backup_20240101.sql
```

### 2. Configuration Backup

```bash
# Backup configuration
tar -czf config_backup.tar.gz .env docker-compose.yml nginx.conf
```

## Troubleshooting

### Common Issues

1. **Database Connection Issues**
   ```bash
   # Check database connectivity
   docker-compose exec postgres psql -U goquant_user -d goquant -c "SELECT 1;"
   ```

2. **Memory Issues**
   ```bash
   # Increase memory limits
   docker-compose up -d --memory=2g goquant
   ```

3. **API Rate Limits**
   ```bash
   # Check rate limit status
   goquant-sentiment status
   ```

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
export DEBUG=true

# Run with debug
goquant-sentiment run
```

## Security Considerations

1. **API Keys**: Store in environment variables, never in code
2. **Database**: Use strong passwords and connection encryption
3. **Network**: Use HTTPS in production
4. **Access**: Implement proper authentication and authorization
5. **Updates**: Keep dependencies updated regularly

## Support

For issues and questions:
1. Check the logs: `docker-compose logs goquant`
2. Review health status: `curl http://localhost/health`
3. Check system resources: `docker stats`
4. Consult the documentation and error messages
