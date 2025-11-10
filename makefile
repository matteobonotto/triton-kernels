

style:
	poetry run black src/triton_kernels

benchmark:
	python -m triton_kernels.benckmarck #--all