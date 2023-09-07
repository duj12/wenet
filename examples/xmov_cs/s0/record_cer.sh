#!/bin/bash
stage=1

if [ $stage -eq 1 ]; then
for z in thchs valid ; do
    for y in farA2; do
        for x in anjing neizao waizao neiwaizao zhengqianfang ceqianfang ;do
            grep $x exp/u2_$y/runtime_16_rw0.3/farfield/${z}_$y/text > exp/u2_$y/runtime_16_rw0.3/farfield/${z}_$y/$x.asr ;
            python3 tools/compute-wer.py --char=1 --v=1 data/farfield/${z}_$y/text exp/u2_$y/runtime_16_rw0.3/farfield/${z}_$y/$x.asr  \
                > exp/u2_$y/runtime_16_rw0.3/farfield/${z}_$y/$x.cer;
            tail  exp/u2_$y/runtime_16_rw0.3/farfield/${z}_$y/$x.cer
        done;
    done;
done
fi

if [ $stage -eq 2 ]; then
for z in test_sets/test_ylfar ; do
    for y in  farA2_nearYL; do
        for x in anjing neizao waizao neiwaizao zhengqianfang ceqianfang ;do
            grep $x exp/u2_$y/ngram/lm_250G_3gram+YouLing3_3gram_chars_with_runtime_16_penalty-3.0_rw0.3/${z}/text \
                  > exp/u2_$y/ngram/lm_250G_3gram+YouLing3_3gram_chars_with_runtime_16_penalty-3.0_rw0.3/${z}/$x.asr ;
            python3 tools/compute-wer.py --char=1 --v=1 data/${z}/text \
                exp/u2_$y/ngram/lm_250G_3gram+YouLing3_3gram_chars_with_runtime_16_penalty-3.0_rw0.3/${z}/$x.asr  \
              > exp/u2_$y/ngram/lm_250G_3gram+YouLing3_3gram_chars_with_runtime_16_penalty-3.0_rw0.3/${z}/$x.cer;
            
            tail exp/u2_$y/ngram/lm_250G_3gram+YouLing3_3gram_chars_with_runtime_16_penalty-3.0_rw0.3/${z}/$x.cer;
        done;
    done;
done
fi