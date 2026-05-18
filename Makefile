.PHONY: preview serve build install help

# Default port for the local dev server
PORT ?= 4000

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-12s %s\n", $$1, $$2}'

install: ## Install Ruby gem dependencies (run once)
	bundle install

preview: ## Local preview with drafts + future posts + livereload (private to localhost)
	bundle exec jekyll serve --drafts --future --livereload \
	  --port $(PORT) --config _config.yml,_config.dev.yml

serve: ## Local preview without drafts (what visitors will see)
	bundle exec jekyll serve --livereload \
	  --port $(PORT) --config _config.yml,_config.dev.yml

build: ## One-off site build into _site/ (no server)
	bundle exec jekyll build --config _config.yml
