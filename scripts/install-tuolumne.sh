ml load python/3.11.5 && python3 -m venv .venvs/scaffoldvenv-tuo && source .venvs/scaffoldvenv-tuo/bin/activate && pip install --upgrade pip
ml cce/21.0.0 cray-mpich/9.1.0 rocm/7.1.0 rccl/fast-env-slows-mpi
pip install -e .[rocmwci] --prefix=.venvs/scaffoldvenv-tuo 2>&1 | tee install.log
# Needed until new wheel exists for torch using mpich 9.1.0
LIB=.venvs/scaffoldvenv-tuo/lib/python3.11/site-packages/torch/lib/libtorch_cpu.so
patchelf --print-needed "$LIB"
patchelf --replace-needed libmpi_gnu_112.so.12 libmpi_gnu.so.12 "$LIB"
patchelf --print-needed "$LIB"
