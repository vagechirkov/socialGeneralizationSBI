

```bash

poetry show --tree  

poetry run pip install --force-reinstall numpy==1.23.5
poetry run python -c "import numpy; print(numpy.__version__)"
```



```bash

curl https://pyenv.run | bash

export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"


# tmux new -s sbi
tmux attach -t sbi

poetry run python3 utils.py --n_groups_simulation 10_000 --prior_type 'weakly_informed' --num_samples 50_000
poetry run python3 utils.py --n_groups_simulation 10_000 --prior_type 'box_uniform' --num_samples 50_000
```