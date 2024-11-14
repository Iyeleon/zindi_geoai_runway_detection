CONFIG_PATH = ./config.toml
.PHONY: train_data
train_data: ./src/data_download.py  $(CONFIG_PATH)
	python3 $< -c $(CONFIG_PATH) -t train -dt 30
.PHONY: test_data
test_data: ./src/data_download.py  $(CONFIG_PATH)
	python3 $< -c $(CONFIG_PATH) -t test -dt 30
.PHONY: run_notebook
run_notebook: ./runway_detection.ipynb
	ipython $<