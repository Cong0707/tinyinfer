import torch, time
from torch.profiler import profile, ProfilerActivity, DeviceType
a = torch.randn(1024,1024,device="cuda"); b = torch.randn(1024,1024,device="cuda")
def fn():
    for _ in range(3): torch.mm(a,b)
fn(); torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CUDA]) as prof:
    for _ in range(4): fn()
    torch.cuda.synchronize()
print("events type:", type(prof.events()))
evs = list(prof.events())
print("num events", len(evs))
for e in evs[:8]:
    print(type(e), getattr(e,"name","?")[:40], getattr(e,"device_type",None), getattr(e,"is_async",None), getattr(e,"cuda_time_total",None), getattr(e,"self_device_time_total",None))
ka = prof.key_averages()
print("key avgs", len(ka))
for e in ka[:5]: print(e.key[:30], e.count, getattr(e,"self_device_time_total",None), getattr(e,"device_time_total",None))
