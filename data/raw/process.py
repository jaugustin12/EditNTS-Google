with open('normal.aligned', encoding="utf8") as f:
    contents = f.readlines()
    # contents = contents[0:25000]
    out_string = ""
    for phrase in contents:
        text = phrase.split('	')
        out_string = text[2]
        with open('normal-processed.txt', 'a', encoding="utf8") as w:
            w.write(out_string)