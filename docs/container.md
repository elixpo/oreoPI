# Container deployment

The Oreo image packages the local daemon and developer CLI for Linux AMD64 and
ARM64. It runs as numeric user `10001`, keeps its root filesystem read-only,
opens no network port, and stores SQLite and redacted session journals only in
the `/var/lib/oreo` volume.

## Laptop workflow

Build and start the daemon:

```bash
docker compose build
docker compose up -d
docker compose ps
```

Exercise the local API and CLI inside the running container:

```bash
docker compose exec oreo elixpo diagnostics
docker compose exec oreo elixpo status
docker compose exec oreo elixpo timer set tea 30
docker compose exec oreo elixpo timer list
docker compose exec oreo elixpo memory list
```

Provider credentials are not stored in the image or Compose file. Export them
in the invoking shell only when testing a live agent turn:

```bash
export POLLINATIONS_API_KEY=...
export OREO_MODEL=...
docker compose exec -e POLLINATIONS_API_KEY -e OREO_MODEL oreo \
  elixpo ask "Hello Oreo"
```

Stop the service while retaining device state:

```bash
docker compose down
```

`docker compose down -v` also deletes the named SQLite/session volume and is
therefore intentionally not part of the normal workflow.

## Multi-architecture image

Create and select a Buildx builder once, then publish both supported Linux
architectures to a registry:

```bash
docker buildx create --name oreo-builder --use
docker buildx inspect --bootstrap
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  --tag <registry>/elixpo/oreopi:<tag> \
  --push .
```

The Dockerfile deliberately compiles once per target platform. There are no
architecture-specific source branches, and the same persistent-state schema is
used on the laptop and SBC.

## Fast definition check

This check does not download images or compile the workspace:

```bash
sh scripts/check-container-definition.sh
```
