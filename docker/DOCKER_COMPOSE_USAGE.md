# SwarmMap Docker Compose Setup

This guide explains how to run SwarmMap server and client in separate Docker containers using Docker Compose.

## Overview

- **Single Image**: Both server and client use the same image: `swarmmap:cuda10.2_with_example`
- **Separate Containers**: Server and client run in isolated containers
- **Docker Network**: Containers communicate via custom bridge network `swarmmap-network`
- **GPU Support**: Both containers have access to NVIDIA GPUs

## Files

- `docker-compose.yml` - Main orchestration file
- `docker/client.yaml` - Client config (HOST points to server container)
- `config/test_mh01.yaml` - Server config (HOST: 0.0.0.0)

## Prerequisites

1. Docker with NVIDIA GPU support (nvidia-docker2)
2. Docker Compose
3. Existing SwarmMap image: `swarmmap:cuda10.2_with_example`
4. EuRoC dataset at: `C:\Users\sj99\Desktop\EuRoC` (Windows)

## Configuration

### Server Configuration
- Uses `config/test_mh01.yaml`
- HOST: `0.0.0.0` (listens on all interfaces)
- PORT: `10088`

### Client Configuration
- Uses `docker/client.yaml`
- HOST: `swarm-server` (Docker DNS resolves to server container)
- PORT: `10088`

## Usage

### 1. Start Both Server and Client

```bash
cd /path/to/SwarmMap
docker-compose up
```

To run in detached mode (background):
```bash
docker-compose up -d
```

### 2. View Logs

All containers:
```bash
docker-compose logs -f
```

Server only:
```bash
docker-compose logs -f swarm-server
```

Client only:
```bash
docker-compose logs -f swarm-client-1
```

### 3. Stop Containers

```bash
docker-compose down
```

### 4. Restart Containers

```bash
docker-compose restart
```

## Individual Container Control

### Start only server:
```bash
docker-compose up swarm-server
```

### Start only client (server must be running):
```bash
docker-compose up swarm-client-1
```

## Accessing Running Containers

### Execute bash in server container:
```bash
docker exec -it swarm-server /bin/bash
```

### Execute bash in client container:
```bash
docker exec -it swarm-client-1 /bin/bash
```

## Output Files

Map and trajectory files are saved to `./output/` directory:
- `map-client-<client_id>.bin`
- `map-server-<client_id>.bin`
- `map-global.bin`
- `KeyFrameTrajectory=<timestamp>-<client_id>.txt`

## Network Details

- **Network Name**: `swarmmap-network`
- **Network Type**: Bridge
- **Server Port**: 10088 (exposed to host)

Containers can communicate using hostnames:
- Server: `swarm-server`
- Client: `swarm-client-1`

## Adding More Clients

Uncomment and configure additional client sections in `docker-compose.yml`:

```yaml
swarm-client-2:
  image: swarmmap:cuda10.2_with_example
  # ... (copy from swarm-client-1 and adjust name)
```

Create corresponding config files:
- `docker/client-2.yaml` with different dataset

## Troubleshooting

### Check if containers are running:
```bash
docker-compose ps
```

### Check GPU access:
```bash
docker exec -it swarm-server nvidia-smi
```

### Network connectivity test:
```bash
# From client container
docker exec -it swarm-client-1 ping swarm-server
```

### Remove everything and start fresh:
```bash
docker-compose down
docker-compose up --force-recreate
```

## Comparison: Single Container vs Docker Compose

### Your Original Single Container Approach:
```bash
docker run --gpus all -it -v C:\Users\sj99\Desktop\EuRoC:/dataset swarmmap:cuda10.2_with_example /bin/bash
# Inside container:
./bin/swarm_server ... &
./bin/swarm_client ...
```

### Docker Compose Approach:
```bash
docker-compose up
```

**Benefits of Docker Compose:**
- Clean separation of server and client
- Easy to scale (multiple clients)
- Automatic network setup
- Better resource isolation
- Easier to manage and monitor

**Same Result:**
- Both approaches use the same image
- Same executables run
- Same dataset access
