# CTFd workshop setup (PwnzzAI)

This directory runs **[CTFd](https://github.com/CTFd/CTFd)** with the **docker_challenges** plugin (**cloned from [northdpole/CTFd-Docker-Challenges](https://github.com/northdpole/CTFd-Docker-Challenges)** — a fork of [offsecginger/CTFd-Docker-Challenges](https://github.com/offsecginger/CTFd-Docker-Challenges) with compatibility fixes — at CTFd image build time), plus a shared **[Ollama](https://ollama.com/)** service and a **PwnzzAI** image that participants spawn per session from the scoreboard.

Setup is **two steps**: (1) bootstrap Docker, the workshop image, and the stack; (2) after the CTFd setup wizard, register the challenge with an **admin API token**. The registration script also **applies Docker API settings** (`docker-socket-proxy:2375`) inside the local `ctfd` container so you do not need to use Admin → Docker Config manually for the default stack.

## What gets installed

| Piece | Role |
|--------|------|
| Docker Engine | Host runtime; the plugin creates per-user containers via the Docker API. |
| `deploy/docker-compose.workshop.yml` | Brings up **Ollama**, **docker-socket-proxy** (restricted Docker API), and **CTFd** (custom image that includes the plugin). |
| Workshop image (`pwnzzai-workshop:latest` by default) | Built from `deploy/Dockerfile.pwnzzai-workshop` — Flask app only; talks to Ollama over the Docker bridge gateway. |

Related files: **`deploy/`** (compose, Dockerfiles), **`deploy/register_pwnzzai_challenge.py`** (API helper used by step 2).

### `docker_challenges` plugin (Git clone)

**`deploy/Dockerfile.ctfd-workshop`** runs **`git clone`** of **[northdpole/CTFd-Docker-Challenges](https://github.com/northdpole/CTFd-Docker-Challenges)** (branch **`master`** by default) during the CTFd image build. Nothing under this repo is copied in — **each deploy pulls the plugin from GitHub**. Override with **`DOCKER_CHALLENGES_GIT_URL`** and **`DOCKER_CHALLENGES_REF`** in **`.env`** to use another repo (for example upstream [offsecginger/CTFd-Docker-Challenges](https://github.com/offsecginger/CTFd-Docker-Challenges)) or pin a tag. After changing those variables, **`./scripts/ctfd_setup/redeploy-ctfd-workshop.sh`** rebuilds CTFd so the new ref is used.

### Alternatives if you do not want that plugin at all

| Approach | Tradeoff |
|----------|----------|
| **Shared lab URL** | One **standard** (non-docker) CTFd challenge whose description points at a **single** PwnzzAI instance (or a small pool managed outside CTFd). Simplest ops; no per-user isolation. |
| **[ctfd-whale](https://github.com/frankli0324/ctfd-whale)** / **[ctfd-challenge-container-plugin](https://github.com/0xfbad/ctfd-challenge-container-plugin)** | Other dynamic-container integrations (different models: FRP, multi-host, etc.). You would replace our compose CTFd image and docs — more moving parts. |
| **[chall-manager](https://github.com/ctfer-io/chall-manager)** | External orchestrator (often K8s-oriented) with CTFd integration; heavier than “compose on one VM”. |

For a **workshop VM** with **one compose file**, the forked plugin is usually the smallest change; switching products is justified when you need multi-tenant orchestration at scale.

### Redeploy CTFd only (after compose or plugin Git settings)

If you already ran the bootstrap and only need to **rebuild the scoreboard** (CTFd) — for example after changing **`DOCKER_CHALLENGES_GIT_URL`** / **`DOCKER_CHALLENGES_REF`** in **`.env`**, or to pull a newer commit from the fork — run from the **repository root**:

```bash
./scripts/ctfd_setup/redeploy-ctfd-workshop.sh
```

The script rebuilds the **`ctfd`** image and restarts that container. It does **not** reinstall Docker, rebuild the PwnzzAI workshop image, or restart Ollama unless you use the full bootstrap again. The repo-root **`.env`** is loaded; **`DOCKER_CHALLENGES_PUBLIC_HOST`** must be set (same rules as bootstrap).

### Teardown (stop CTFd, Ollama, and the workshop stack)

To **stop and remove** the compose services (**`ctfd`**, **`ollama`**, **`docker-socket-proxy`**):

```bash
./scripts/ctfd_setup/teardown-ctfd-workshop.sh
```

- **Default:** containers are removed but **named volumes are kept** (your CTFd database and Ollama model files remain for the next `up`).
- **Delete that data too:** add **`--volumes`** (or **`-v`**). This wipes **`ctfd_uploads`** (CTFd DB, uploads) and **`ollama_data`** (downloaded models).
- **Remove the PwnzzAI app image** built by bootstrap (tag **`pwnzzai-workshop:latest`** by default): add **`--rmi-workshop`**.

Participant **challenge containers** (per-user PwnzzAI instances) are normal Docker containers on the host; they are **not** removed by compose down. After teardown, check with **`docker ps -a`** and remove stragglers if you want a clean host (**`docker stop`** / **`docker rm`**).

## Step 1 — Bootstrap (VM / first install)

From the **repository root**:

```bash
cp .env.example .env
# Edit .env: set DOCKER_CHALLENGES_PUBLIC_HOST to this machine's public hostname or IP (required).
chmod +x scripts/ctfd_setup/bootstrap-ctfd-workshop.sh
./scripts/ctfd_setup/bootstrap-ctfd-workshop.sh
```

**Required:** a repo-root **`.env`** with **`DOCKER_CHALLENGES_PUBLIC_HOST`** (see **`.env.example`**). Bootstrap, redeploy, **`docker compose`** for CTFd, and the register scripts will fail without it. You can use **`PWNZZAI_PUBLIC_HOST`** instead; scripts treat it as an alias.

When it finishes, open **`http://<your-host>:8000`** and complete the **CTFd setup wizard** if this is the first run.

### Docker API in CTFd (default stack)

If you use **`deploy/docker-compose.workshop.yml`** on this machine, **step 2’s** `register_pwnzzai_challenge.py` updates the **`docker_config`** row in CTFd’s database inside the **`ctfd`** container (hostname **`docker-socket-proxy:2375`**, TLS off). You need the **Docker CLI** and a running **`ctfd`** service. Override with **`CTFD_DOCKER_API_HOST`**; skip with **`CTFD_SKIP_DOCKER_CONFIG=1`** (then set **Admin → Docker Config** yourself, or for a remote CTFd instance).

### Get an admin API token (required for step 2)

1. **Admin → Settings → API Tokens**
2. Generate a token and copy it (you will not be shown the full value again in some CTFd versions).

## Step 2 — Register the PwnzzAI challenge

After step 1 and the CTFd wizard, run:

```bash
chmod +x scripts/ctfd_setup/register-pwnzzai-challenge.sh
export CTFD_URL='http://127.0.0.1:8000'   # change if CTFd is elsewhere
export CTFD_API_TOKEN='paste-token-here'
# Optional:
# export CHALLENGE_FLAG='flag{example}'
./scripts/ctfd_setup/register-pwnzzai-challenge.sh
```

You can put **`CTFD_API_TOKEN`** (and optionally **`CTFD_URL`**, **`CHALLENGE_FLAG`**) in **`.env`** instead of exporting in the shell; the script sources the repo-root `.env` if present.

`CTFD_API_KEY` is accepted as an alias for **`CTFD_API_TOKEN`**.

**Manual alternative:** **Admin → Challenges → New** → type **docker** → set the image tag to **`pwnzzai-workshop:latest`** (or the tag you built via `PWNZZAI_WORKSHOP_IMAGE`).

Participants should then see **Start** / spawn controls on that challenge, depending on the plugin UI.

## Environment variables and `.env`

### The important caveat

The Docker Challenges plugin starts containers from an **image name and tag**. It does **not** inject your `.env` at container start. The bootstrap script **bakes** supported variables into the image **at build time** using `deploy/Dockerfile.pwnzzai-workshop`. Every spawned instance shares that configuration.

If you need **different secrets per participant**, you need a different approach (for example a plugin that supports runtime env, or a custom orchestrator).

### Variables used when building the workshop image

If present in `.env` (or the environment when you run the bootstrap), these are passed as Docker build arguments:

- `SECRET_KEY` — PwnzzAI Flask secret
- `GEMINI_API_KEY`, `GOOGLE_API_KEY`, `GEMINI_MODEL` — optional Gemini routing
- `OLLAMA_HOST` — URL the app uses to reach Ollama (see next section)
- `OLLAMA_FALLBACK_MODEL`

If **`OLLAMA_HOST`** is unset, the bootstrap picks a default aimed at containers on the default Docker bridge, typically **`http://<docker0-gateway>:11434`** or **`http://172.17.0.1:11434`**, so user instances can reach Ollama published on the host.

### CTFd itself

- **`DOCKER_CHALLENGES_PUBLIC_HOST`** (or **`PWNZZAI_PUBLIC_HOST`**) — **required** — public hostname or IP shown in challenge links; passed into the **`ctfd`** container (see [Networking and cloud security](#networking-and-cloud-security)).
- **`DOCKER_CHALLENGES_GIT_URL`** — Git URL for the **docker_challenges** plugin (default **`https://github.com/northdpole/CTFd-Docker-Challenges.git`**).
- **`DOCKER_CHALLENGES_REF`** — branch or tag to clone (default **`master`**).
- **`CTFD_SECRET_KEY`** — forwarded into the CTFd container as `SECRET_KEY` (recommended in production).
- **`PWNZZAI_ROOT`** — if the repo is not two levels above the script, set this to the repo root explicitly.
- **`PWNZZAI_WORKSHOP_IMAGE`** — override the tag used for the workshop image (default `pwnzzai-workshop:latest`).

### Variables for step 2 (register script)

- **`DOCKER_CHALLENGES_PUBLIC_HOST`** — **required** in **`.env`** (unless **`ALLOW_CHALLENGE_REGISTER_WITHOUT_PUBLIC_HOST=1`** for rare setups).
- **`CTFD_URL`** — CTFd base URL (default `http://127.0.0.1:8000`).
- **`CTFD_API_TOKEN`** or **`CTFD_API_KEY`** — admin token (required).
- **`CTFD_SKIP_DOCKER_CONFIG`** — set to `1` to skip automatic Docker hostname injection (remote CTFd or manual **Admin → Docker Config**).
- **`CTFD_DOCKER_API_HOST`** — Docker API hostname for the plugin (default `docker-socket-proxy:2375`).
- **`CTFD_COMPOSE_FILE`** — compose file path relative to repo root (default `deploy/docker-compose.workshop.yml`).
- **`PWNZZAI_IMAGE`** — override docker image tag for the challenge (defaults to `PWNZZAI_WORKSHOP_IMAGE` or `pwnzzai-workshop:latest`).
- **`CHALLENGE_NAME`**, **`CHALLENGE_FLAG`** — passed through to `deploy/register_pwnzzai_challenge.py`.

You can also run **`python3 deploy/register_pwnzzai_challenge.py`** directly with the same environment variables.

### Re-register (delete old challenge, create new)

To remove an existing challenge with the same **`CHALLENGE_NAME`** and create a fresh one (fixes bad settings or duplicate):

```bash
export CTFD_URL='http://127.0.0.1:8000'
export CTFD_API_TOKEN='...'
./scripts/ctfd_setup/reregister-pwnzzai-challenge.sh
```

Or: **`python3 deploy/reregister_pwnzzai_challenge.py`** (same env vars as registration).

## Networking and cloud security

- **Port 8000** — CTFd HTTP (put HTTPS in front with a reverse proxy for production).
- **Port 11434** — Ollama is published for access from per-user containers via the host bridge. On a public instance, **restrict** this port in your security group or firewall unless you intend to expose Ollama broadly.
- **Challenge links (`Host:` / `http://…`) — required `DOCKER_CHALLENGES_PUBLIC_HOST`** — In **Admin → Docker Config** you set **`docker-socket-proxy:2375`** so CTFd can talk to Docker. That name is **only valid inside** the compose network; participants need this server’s **public hostname or IP** in **`.env`** as **`DOCKER_CHALLENGES_PUBLIC_HOST`** (copy **`.env.example`**). The stack will not start CTFd via compose without it, and register scripts refuse to create the challenge without it. After changing **`.env`**, run **`./scripts/ctfd_setup/redeploy-ctfd-workshop.sh`**. Participants open **`http://<that-host>:<port>`** using the port shown on the challenge.

The stack uses **[tecnativa/docker-socket-proxy](https://github.com/Tecnativa/docker-socket-proxy)** so CTFd does not mount the raw Docker socket. Treat the host as **privileged** anyway: participants run arbitrary code in containers on this machine.

## EC2 user-data example

Run the bootstrap after cloning the repo and placing `.env` (or fetching secrets from your preferred store):

```bash
#!/bin/bash
set -euo pipefail
exec > /var/log/pwnzzai-ctfd-bootstrap.log 2>&1
apt-get update && apt-get install -y git
git clone https://github.com/<your-org>/PwnzzAI.git /opt/PwnzzAI
# cp /path/to/.env /opt/PwnzzAI/.env
chmod +x /opt/PwnzzAI/scripts/ctfd_setup/bootstrap-ctfd-workshop.sh
/opt/PwnzzAI/scripts/ctfd_setup/bootstrap-ctfd-workshop.sh
# After you complete the CTFd wizard and create an API token on the instance:
#   export CTFD_API_TOKEN='...'
#   /opt/PwnzzAI/scripts/ctfd_setup/register-pwnzzai-challenge.sh
```

Adjust package installation and paths for **Amazon Linux** or other AMIs if you are not on Debian/Ubuntu.

## Troubleshooting

- **Docker Config cannot connect** — Confirm the hostname is exactly what the CTFd container can resolve (for example `docker-socket-proxy:2375`) and that the compose stack is running: `docker compose -f deploy/docker-compose.workshop.yml ps`.
- **Internal Server Error when starting a challenge container** — This is almost always the **CTFd app** crashing while talking to Docker (uncaught Python exception), not the registration script.
  1. **Plugin + image rebuild** — The CTFd image **clones** the plugin from GitHub at build time (northdpole fork by default). To pick up fork changes or change **`DOCKER_CHALLENGES_REF`**, run **`./scripts/ctfd_setup/redeploy-ctfd-workshop.sh`**, or manually: `cd deploy` then **`docker compose -f docker-compose.workshop.yml build ctfd`** and **`... up -d --force-recreate ctfd`**.
  2. **Socket proxy ACLs** — The Tecnativa proxy needs **`ALLOW_START=1`** (and related flags). The repo `deploy/docker-compose.workshop.yml` sets these; **`docker compose up -d --force-recreate docker-socket-proxy`** if you changed the file.
  3. **Docker hostname in CTFd** — **`docker-socket-proxy:2375`** only (no **`http://`**).
  4. **Tagged image name** — Use **`pwnzzai-workshop:latest`** (with **`:tag`**). A name without a tag can crash the plugin.
  5. **Image on the host** — The daemon must have the image: `docker images | grep pwnzzai`. Build with the bootstrap script or `docker build -f deploy/Dockerfile.pwnzzai-workshop -t pwnzzai-workshop:latest ..` from repo root.
  6. **Quick probe** — With the stack running: **`deploy/debug-ctfd-docker.sh`** (checks version + image inspect from inside the CTFd container).
  7. **Logs** — `docker compose -f deploy/docker-compose.workshop.yml logs -f ctfd` — look for **`KeyError: 'Id'`**, **`401`/`403`** from the proxy, or tracebacks in **`docker_challenges`**.
- **Participants’ apps cannot reach Ollama** — Check `OLLAMA_HOST` baked into the image, firewall rules, and that Ollama is listening on the host port expected from the bridge gateway.
- **Challenge registration fails** — Complete the CTFd setup wizard first; ensure the API token is valid and `CTFD_URL` matches where CTFd listens. If registration fails on `docker compose exec`, set **`CTFD_SKIP_DOCKER_CONFIG=1`** and configure **Admin → Docker Config** manually.

## Files in this directory

| File | Purpose |
|------|---------|
| `bootstrap-ctfd-workshop.sh` | Step 1: Docker, workshop image, compose stack. Prints instructions for step 2. |
| `redeploy-ctfd-workshop.sh` | Rebuild and restart only the CTFd container (e.g. after changing plugin Git URL/ref in `.env`). |
| `teardown-ctfd-workshop.sh` | Stop and remove the compose stack; optional `--volumes`, `--rmi-workshop`. |
| `register-pwnzzai-challenge.sh` | Step 2: Creates the PwnzzAI docker challenge via CTFd API (needs admin token). |
| `reregister-pwnzzai-challenge.sh` | Deletes challenges matching `CHALLENGE_NAME`, then creates the challenge again. |
| `README.md` | This document. |

Python helpers under **`deploy/`**: `register_pwnzzai_challenge.py`, `reregister_pwnzzai_challenge.py`, `debug-ctfd-docker.sh` (connectivity + image inspect). The **docker_challenges** plugin is **not** vendored here; it is cloned from **`DOCKER_CHALLENGES_GIT_URL`** when the CTFd image is built.

Compose and image definitions remain in **`deploy/`** at the repository root.
