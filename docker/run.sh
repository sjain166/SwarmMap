#!/bin/bash

# SwarmMap Docker Compose Helper Script
# Runs server and client containers using docker-compose

set -e

echo "=========================================="
echo "SwarmMap Docker Compose Runner"
echo "=========================================="

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    echo "Error: docker-compose is not installed"
    exit 1
fi

# Go to project root
cd "$(dirname "$0")/.."

# Function to show usage
usage() {
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  up       - Start server and client containers"
    echo "  down     - Stop and remove containers"
    echo "  logs     - Show logs from all containers"
    echo "  server   - Show server logs only"
    echo "  client   - Show client logs only"
    echo "  restart  - Restart all containers"
    echo "  build    - Build/rebuild the Docker image"
    echo ""
    exit 1
}

# Parse command
case "${1:-up}" in
    up)
        echo "Starting SwarmMap containers..."
        docker-compose up -d
        echo ""
        echo "Containers started successfully!"
        echo "Server: swarm-server"
        echo "Client: swarm-client-1"
        echo ""
        echo "To view logs, run: docker-compose logs -f"
        ;;

    down)
        echo "Stopping SwarmMap containers..."
        docker-compose down
        echo "Containers stopped."
        ;;

    logs)
        echo "Showing logs from all containers (Ctrl+C to exit)..."
        docker-compose logs -f
        ;;

    server)
        echo "Showing server logs (Ctrl+C to exit)..."
        docker-compose logs -f swarm-server
        ;;

    client)
        echo "Showing client logs (Ctrl+C to exit)..."
        docker-compose logs -f swarm-client-1
        ;;

    restart)
        echo "Restarting SwarmMap containers..."
        docker-compose restart
        echo "Containers restarted."
        ;;

    build)
        echo "Building Docker image..."
        echo "Make sure your Dockerfile builds an image tagged as 'swarmmap:latest'"
        echo ""
        docker build -t swarmmap:latest -f Dockerfile .
        echo ""
        echo "Image built successfully!"
        ;;

    *)
        usage
        ;;
esac
