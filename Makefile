.PHONY: preview serve build install help

# Default port for the local dev server
PORT ?= 4000

# jekyll-sass-converter 1.5.2 (shipped with jekyll 3.10) breaks on UTF-8 SCSS
# files unless the default locale is UTF-8. Force it for every target so
# you don't have to remember.
export LANG := en_US.UTF-8
export LC_ALL := en_US.UTF-8

# Use a project-local gem dir so we never need sudo to write to /usr/local.
BUNDLE_FLAGS := --path vendor/bundle

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-12s %s\n", $$1, $$2}'

install: ## Install Ruby gem dependencies into vendor/bundle (run once)
	bundle config set --local path 'vendor/bundle'
	bundle install

preview: ## Local preview with drafts + future posts + livereload (private to localhost)
	bundle exec jekyll serve --drafts --future --livereload \
	  --port $(PORT) --config _config.yml,_config.dev.yml

serve: ## Local preview without drafts (what visitors will see)
	bundle exec jekyll serve --livereload \
	  --port $(PORT) --config _config.yml,_config.dev.yml

build: ## One-off site build into _site/ (no server)
	bundle exec jekyll build --config _config.yml
