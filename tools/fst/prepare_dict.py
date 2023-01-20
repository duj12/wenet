#!/usr/bin/env python3
# encoding: utf-8

import sys

def is_Chinese(word):
    for ch in word:
        if '\u4e00' <= ch <= '\u9fff':
            return True
    return False

# sys.argv[1]: e2e model unit file(lang_char.txt)
# sys.argv[2]: raw lexicon file
# sys.argv[3]: output lexicon file
# sys.argv[4]: bpemodel
# sys.argv[5]: chinese_unit_is_chars

unit_table = set()
with open(sys.argv[1], 'r', encoding='utf8') as fin:
    for line in fin:
        unit = line.split()[0]
        unit_table.add(unit)


def contain_oov(units):
    for unit in units:
        if unit not in unit_table:
            return True
    return False

chinese_unit_is_chars=0
bpemode = len(sys.argv) > 4
if bpemode:
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.Load(sys.argv[4])
    if len(sys.argv) > 5:
        chinese_unit_is_chars=int(sys.argv[5])
lexicon_table = set()
with open(sys.argv[2], 'r', encoding='utf8') as fin, \
        open(sys.argv[3], 'w', encoding='utf8') as fout:
    for line in fin:
        word = line.split()[0]
        if word == 'SIL' and not bpemode:  # `sil` might be a valid piece in bpemodel
            continue
        elif word == '<SPOKEN_NOISE>':
            continue
        else:
            # each word only has one pronunciation for e2e system
            if word in lexicon_table:
                continue
            if bpemode:
                if is_Chinese(word) and chinese_unit_is_chars:
                    chars = ' '.join([char for char in word])
                    pieces = sp.EncodeAsPieces(chars)
                else:
                    pieces = sp.EncodeAsPieces(word)
                if contain_oov(pieces):
                    print(
                        'Ignoring words {}, which contains oov unit'.format(
                            ''.join(word).strip('▁'))
                    )
                    continue
                chars = ' '.join(
                    [p if p in unit_table else '<unk>' for p in pieces])
            else:
                # ignore words with OOV
                if contain_oov(word):
                    print('Ignoring words {}, which contains oov unit'.format(word))
                    continue
                # Optional, append ▁ in front of english word
                # we assume the model unit of our e2e system is char now.
                if word.encode('utf8').isalpha() and '▁' in unit_table:
                    word = '▁' + word
                chars = ' '.join(word)  # word is a char list
            fout.write('{} {}\n'.format(word, chars))
            lexicon_table.add(word)
