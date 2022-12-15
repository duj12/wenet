export WENET_DIR=$PWD/../../..
export BUILD_DIR=${WENET_DIR}/runtime/libtorch/build
export OPENFST_PREFIX_DIR=${BUILD_DIR}/../fc_base/openfst-subbuild/openfst-populate-prefix
export PATH=$PWD:${BUILD_DIR}/bin:${BUILD_DIR}/kaldi:${OPENFST_PREFIX_DIR}/bin:$PATH

# NOTE(kan-bayashi): Use UTF-8 in Python to avoid UnicodeDecodeError when LC_ALL=C
export PYTHONIOENCODING=UTF-8
export PYTHONPATH=../../../:$PYTHONPATH

CODE_DIR=/data/megastore/Projects/DuJing/code
KALDI_ROOT=$CODE_DIR/kaldi
FAIRSEQ_ROOT=$CODE_DIR/fairseq
S3PRL_ROOT=$CODE_DIR/s3prl
WENET_ROOT=$CODE_DIR/wenet

#[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
#. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C

#. "${MAIN_ROOT}"/tools/activate_python.sh && . "${MAIN_ROOT}"/tools/extra_path.sh

export PYTHONPATH=${MAIN_ROOT}:${FAIRSEQ_ROOT}:${S3PRL_ROOT}:\
$WENET_ROOT/tools/k2/icefall\
${PYTHONPATH:-}

export PATH=$KALDI_ROOT/src/fstbin:$KALDI_ROOT/src/lmbin:\
$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/tools/sph2pipe:\
$KALDI_ROOT/tools/sctk/bin:$KALDI_ROOT/tools/srilm/bin:\
$KALDI_ROOT/tools/srilm/bin/i686-m64:$PATH

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/xmov/anaconda3/envs/fairseq/lib:/home/xmov/miniconda3/envs/fairseq/lib
