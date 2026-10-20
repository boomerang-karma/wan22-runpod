PY ?= python3
IMAGE ?= $(shell $(PY) -c "import yaml;print(yaml.safe_load(open('config.yaml'))['image']['name'])" 2>/dev/null)

.PHONY: setup test mock build deploy prepare info health costs teardown portal portal-mock aws-deploy aws-destroy modal-deploy modal-prepare modal-info

setup:            ## client deps (requests, pyyaml, pillow)
	$(PY) -m pip install -r requirements-client.txt

test:             ## CPU tests: tiny real Wan pipeline, prepare(), full-size LoRA key map, mock handler, client
	$(PY) -m pip install -q -r worker/requirements.txt -r portal/requirements.txt torch pytest httpx && $(PY) -m pytest -q tests

mock:             ## run the worker locally in mock mode through the RunPod SDK test runner
	cd worker && WORKER_MODE=mock VOLUME_PATH=../.mock_volume $(PY) handler.py --test_input '{"input":{"action":"prepare"}}'
	cd worker && WORKER_MODE=mock VOLUME_PATH=../.mock_volume $(PY) handler.py --test_input '{"input":{"action":"info"}}'

build:            ## build + push the worker image for linux/amd64 (needs docker buildx; ~8-10 GB)
	@test -n "$(IMAGE)" || (echo "set image.name in config.yaml" && exit 1)
	docker buildx build --platform linux/amd64 -t $(IMAGE) --push worker

deploy:  ; $(PY) wanctl.py deploy
prepare: ; $(PY) wanctl.py prepare
info:    ; $(PY) wanctl.py info
health:  ; $(PY) wanctl.py health
costs:   ; $(PY) wanctl.py costs
teardown:; $(PY) wanctl.py teardown
portal:      ## Meme Studio web app on http://127.0.0.1:8000 (real endpoint)
	$(PY) portal/server.py
portal-mock: ## Meme Studio without a GPU (fake jobs, no cost)
	PORTAL_MOCK=1 PORTAL_DATA=.mock_portal $(PY) portal/server.py
aws-deploy:  ## Meme Studio on AWS with Auth0 login (settings in deploy/aws/deploy.env)
	deploy/aws/deploy.sh
aws-destroy: ## delete the AWS stack (keeps the gallery's EFS)
	deploy/aws/destroy.sh
modal-deploy:  ## LTX worker on Modal (no GPU time until used)
	$(PY) -m modal deploy deploy/modal/ltx_app.py
modal-prepare: ## once: weights onto the Modal Volume (CPU, ~20-40 min)
	$(PY) -m modal run deploy/modal/ltx_app.py::prepare
modal-info:    ## start a GPU container and report what it loaded (billed)
	$(PY) -m modal run deploy/modal/ltx_app.py::info
