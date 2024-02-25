# Binary search of word in list lword
def bsearch(word,lword):
    min=0
    max=len(lword)
    cur=int(max/2)
    while max>0:
        cur=min+int((max-min)/2)
        if word<lword[cur]:
            max=cur
        else:
            min=cur
        if cur<len(lword)-1:
            if word>=lword[cur] and word<lword[cur+1]:
                break
        else:
            if word>=lword[cur]:
                break
        if max==0:
            cur=-1
    return cur

def _binsert(word,lword):
    cur=bsearch(word,lword)
    return binsert(word,lword,cur)

# Insert word in list lword at position cur
def binsert(word,lword,cur):
    if cur>=0:
        if cur<len(lword)-1:
            lword=lword[0:cur+1]+[word]+lword[cur+1:len(lword)]
        else:
            lword=lword[0:cur+1]+[word]
    else:
        if cur<len(lword)-1:
            lword=[word]+lword[0:len(lword)]
        else:
            lword=[word]
    return lword