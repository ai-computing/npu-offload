import json,sys
d=json.load(open(sys.argv[1]))
txt={}
for line in d['text'].splitlines():
    if line.startswith('[@'):
        k=line[1:line.index(']')]; txt[k]=line[line.index(']')+2:]
t=d['timeline']['operators']
for k in sorted(t,key=lambda k:int(k[1:])):
    for e in t[k]:
        r=e['resources']
        if r==['Latency'] : continue
        b,en=e['lifetime']['begin'],e['lifetime']['end']
        if en-b<50 and 'Dma' not in str(r): continue
        print(f"{k:>5} {b:>8}-{en:>8} ({en-b:>7}) {','.join(r):<26} {txt.get(k,'')[:110]}")
