import math, itertools
from trader import technicals, analysis

def bars_from(closes):
    out=[]
    for c in closes:
        out.append({"open": c, "high": c*1.004, "low": c*0.996, "close": c, "volume": 100000})
    return out

best=None
for down,up,drop,rise in itertools.product([250,270,280],[8,12,20,30],[0.0015,0.003],[0.002,0.004,0.008]):
    closes=[100.0]
    for i in range(down): closes.append(closes[-1]*(1-drop))
    for i in range(up):   closes.append(closes[-1]*(1+rise))
    b=bars_from(closes[-300:])
    t=technicals.technical_signal(b)
    if t["signal"]=="BUY":
        sig={"technical":t,"sentiment":{"signal":"HOLD","strength":0.0,"reason":"x"},"dividend":{"signal":"HOLD","strength":0.0,"reason":"y"}}
        f=analysis.fuse(sig)
        print(down,up,drop,rise,"->",t["signal"],t["strength"],"conf",f["confidence"],"price",round(t['current_price'],2),"rsi",round(t['indicators']['rsi'],1))
