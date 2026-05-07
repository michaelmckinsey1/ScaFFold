ml load python/3.12.2 && python3 -m venv .venvs/scaffoldvenv-tuo-pypi && source .venvs/scaffoldvenv-tuo-pypi/bin/activate && pip install --upgrade pip
ml cce/21.0.0 cray-mpich/9.1.0 rocm/7.2.1 rccl/fast-env-slows-mpi
pip install -e .[rocm] --find-links https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/ 2>&1 | tee install.log
# libmpi.so.12 does not exist => ls /opt/cray/pe/lib64/ | grep libmpi
patchelf --replace-needed libmpi.so.12 libmpi_gnu.so.12 .venvs/scaffoldvenv-tuo-pypi/lib/python3.12/site-packages/mpi4py/MPI.mpich.cpython-312-x86_64-linux-gnu.so
