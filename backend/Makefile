# ── Fraud Detection — Docker shortcuts ───────────────────────────────────────
.PHONY: build up down restart logs ps shell-api shell-db clean nuke

## Build all images
build:
	docker compose build

## Start all services (detached)
up:
	docker compose up -d

## Start with live logs
up-logs:
	docker compose up

## Stop all services
down:
	docker compose down

## Restart a single service  e.g. make restart svc=api
restart:
	docker compose restart $(svc)

## Tail logs for all services
logs:
	docker compose logs -f

## Tail logs for one service  e.g. make log svc=worker-idcard
log:
	docker compose logs -f $(svc)

## Show running containers + health
ps:
	docker compose ps

## Open shell in API container
shell-api:
	docker compose exec api bash

## Open psql inside postgres container
shell-db:
	docker compose exec postgres psql -U fraud -d fraud_detect

## Remove stopped containers + dangling images (keeps volumes)
clean:
	docker compose down --remove-orphans
	docker image prune -f

## Full wipe — containers, images AND volumes (⚠ deletes DB data)
nuke:
	docker compose down -v --remove-orphans
	docker image prune -af
