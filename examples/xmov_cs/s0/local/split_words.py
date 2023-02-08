#!/usr/bin/env python3

from preprocess import (
    remove_space_between_mandarin,
    remove_redundant_whitespaces,
)
import pkuseg  #目前pkuseg还有分词bug，会分出词表中不存在的词。


if __name__ == '__main__':
    import sys
    input = sys.argv[1]
    if input=="-":
        fin=sys.stdin
    else:
        fin = open(input, 'r', encoding='utf-8')
    output = sys.argv[2]
    if output=='-':
        fout = sys.stdout
    else:
        fout = open(output, 'w', encoding='utf-8')

    user_dict = sys.argv[3]
    has_name = 0
    if len(sys.argv) > 4:
        has_name = int(sys.argv[4])

    seg = pkuseg.pkuseg(user_dict=user_dict)

    for line in fin:
        if has_name:
            line = line.strip().split(' ')
            name = line[0]
            text = ' '.join(line[1:])
            new_text = remove_space_between_mandarin(text)
            new_text = ' '.join(seg.cut(new_text))
            new_text = remove_redundant_whitespaces(new_text)
            fout.write(name + ' ' + new_text + '\n')
        else:
            text = line.strip()
            new_text = remove_space_between_mandarin(text)
            new_text = ' '.join(seg.cut(new_text))
            new_text = remove_redundant_whitespaces(new_text)
            fout.write(new_text + '\n')

    fin.close()
    fout.close()