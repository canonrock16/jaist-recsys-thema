# jaist-recsys-thema
jaistでのレコメンド研究用リポジトリ

# MIND

```shell
cd jaist-recsys-thema
poetry run python -m src.MIND.main
```

# RPCD
```shell
cd jaist-recsys-thema
poetry run python -m src.RPCD.main
```

# kagayaki
```shell
qsub -I -q GPU-1A
setenv PATH ${PATH}:${HOME}/.poetry/bin
poetry run pyenv exec jupyter lab
poetry run pyenv exec python -m src.MIND.main
poetry run python -m src.MIND.main

```

qsub MIND_PBS.sh
