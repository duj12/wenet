#!/usr/bin/env python3

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
    word_set = set()
    with open(user_dict, 'r') as f_dict:
        for line in f_dict:
            word = line.strip().split(' ')[0]
            #print(word)
            word_set.add(word)
    has_name = 0
    if len(sys.argv) > 4:
        has_name = int(sys.argv[4])

    for path in fin:
        path = path.strip()
        print(path)
        f = open(path, 'r', encoding='utf-8')
        for line in f:
            if has_name:
                line = line.strip().split(' ')
                name = line[0]
                text = line[1:]
                for word in text:
                    if not word in word_set:
                        print(f"oov is added: {word}")
                        word_set.add(word)
            else:
                text = line.strip()
                for word in text.split():
                    if not word in word_set:
                        print(f"oov is added: {word}")
                        word_set.add(word)
        f.close()
    sort_words=sorted(list(word_set))
    for word in sort_words:
        #print(word)
        if len(word.strip())>0:
            fout.write(word+"\n")

    fin.close()
    fout.close()