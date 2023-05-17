@echo off
setlocal enabledelayedexpansion
set workspace=C:\Users\Samuel\Desktop\BP_proj\BP-NN_Visualisation
set program=%workspace%\src\__main__.py
set PYTHONPATH=%workspace%\src;%PYTHONPATH%

python %program% --surface_all --NNmodel 1 --dataset 0 --dropout 0
python %program% --surface_avg --NNmodel 1 --dataset 0 --dropout 1
python %program% --surface_avg --NNmodel 1 --dataset 1 --dropout 0
python %program% --surface_avg --adam --NNmodel 1 --dataset 1 --dropout 0

for /L %%i in (1,1,8) do (
    echo Running iteration %%i
    if %%i LEQ 4 (
        set adam_arg=
    ) else (
        set adam_arg=--adam
    )
    set /a "dataset_arg = (%%i - 1) / 2 %% 2"
    set /a "dropout_arg = (%%i - 1) %% 2"
    python %program% --train --step !adam_arg! --NNmodel 1 --dataset !dataset_arg! --dropout !dropout_arg!
)

for /L %%i in (1,1,8) do (
    echo Running iteration %%i
    if %%i LEQ 4 (
        set adam_arg=
    ) else (
        set adam_arg=--adam
    )
    set /a "dataset_arg = (%%i - 1) / 2 %% 2"
    set /a "dropout_arg = (%%i - 1) %% 2"
    python %program% --train --step !adam_arg! --NNmodel 3 --dataset !dataset_arg! --dropout !dropout_arg!
)

for /L %%i in (1,1,8) do (
    echo Running iteration %%i
    if %%i LEQ 4 (
        set adam_arg=
    ) else (
        set adam_arg=--adam
    )
    set /a "dataset_arg = (%%i - 1) / 2 %% 2"
    set /a "dropout_arg = (%%i - 1) %% 2"
    python %program% --train --step !adam_arg! --NNmodel 4 --dataset !dataset_arg! --dropout !dropout_arg!
)

for /L %%i in (1,1,8) do (
    echo Running iteration %%i
    if %%i LEQ 4 (
        set adam_arg=
    ) else (
        set adam_arg=--adam
    )
    set /a "dataset_arg = (%%i - 1) / 2 %% 2"
    set /a "dropout_arg = (%%i - 1) %% 2"
    python %program% --train --step !adam_arg! --NNmodel 2 --dataset !dataset_arg! --dropout !dropout_arg!
)