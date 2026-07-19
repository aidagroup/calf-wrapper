# CALF-Wrapper tracking stack

This stack is dedicated to CALF-Wrapper and does not share containers, ports,
storage, databases, buckets, or networks with CALF-Enhance.

On `gor`, create the untracked configuration and start the services:

```bash
cp infra/.env.example infra/.env
# Replace both example passwords in infra/.env.
docker compose --env-file infra/.env -f infra/docker-compose.yml config --quiet
docker compose --env-file infra/.env -f infra/docker-compose.yml up -d --build
docker compose --env-file infra/.env -f infra/docker-compose.yml ps
```

The server itself uses `http://127.0.0.1:5001` as the tracking URI. Other
machines on the private network use `http://192.168.1.5:5001`. MinIO is
available at ports `9030` (S3 API) and `9031` (console); experiment clients use
MLflow's artifact proxy and therefore do not need MinIO credentials.

The real `infra/.env` and persistent `.dockerdata/` directory are ignored by
Git. The example credentials are deliberately unusable placeholders.
