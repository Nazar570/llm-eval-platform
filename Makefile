COMPOSE := docker compose -f infrastructure/docker-compose.yml

.PHONY: up down restart logs ps build migrate format lint typecheck test check

up:
	$(COMPOSE) up --build -d

down:
	$(COMPOSE) down

restart:
	$(COMPOSE) down
	$(COMPOSE) up --build -d

logs:
	$(COMPOSE) logs -f

ps:
	$(COMPOSE) ps

build:
	$(COMPOSE) build

migrate:
	EVAL_DATABASE_HOST=localhost \
	EVAL_DATABASE_PORT=5433 \
	EVAL_DATABASE_USER=eval \
	EVAL_DATABASE_PASSWORD=eval \
	EVAL_DATABASE_NAME=evaldb \
	PYTHONPATH=. \
	alembic -c db/alembic.ini upgrade head

format:
	ruff check --fix .
	black .

lint:
	ruff check .
	black --check .

typecheck:
	mypy .

test:
	pytest

check: lint typecheck test