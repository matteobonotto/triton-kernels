

test:
	poetry run pytest -m "not slow" -vs

style:
	poetry run black src/triton_kernels

benchmark:
	python -m triton_kernels.benckmarck #--all

