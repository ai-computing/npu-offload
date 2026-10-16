import json,sys
d=json.load(open(sys.argv[1]))
t=d['timeline']['operators']
rows=[]
for k,v in t.items():
    for e in v:
        r=e['resources']
        if r==['Latency'] or r==['PeCore']: continue
        b,en=e['lifetime']['begin'],e['lifetime']['end']
        rows.append((b,en,k,','.join(r)))
rows.sort()
end=max(r[1] for r in rows)
print("total est cycles:", end, "ops:", len(rows))
for r in rows[:int(sys.argv[2]) if len(sys.argv)>2 else 40]:
    print(f"{r[2]:>14} {r[0]:>8}-{r[1]:>8} ({r[1]-r[0]:>6}) {r[3]}")
