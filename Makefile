.PHONY: install run

install:
	poetry install --no-root

run:
	poetry run python src/spotify_clustering/main.py