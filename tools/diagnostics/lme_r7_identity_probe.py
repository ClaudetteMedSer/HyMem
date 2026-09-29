import sys,json,hashlib,types,random,time
sys.path.insert(0,sys.argv[1])
from hymem.extraction import producer
rng=random.Random(20260925)
out=[]
def wrap(value):
 def reader(): return value
 return reader
for index in range(48):
 shared=[rng.randrange(1000),"é\n",{"values":{1,2,3}}]
 graph=[shared,shared]
 for depth in range(index%4): graph=[graph,graph,(depth,shared)]
 reader=wrap(graph)
 first=producer.canonical_callable_sha256(reader)
 assert producer.canonical_callable_sha256(reader)==first
 shared.append("changed")
 changed=producer.canonical_callable_sha256(reader)
 assert changed!=first
 shared.pop()
 assert producer.canonical_callable_sha256(reader)==first
 out.extend([first,changed])
module=types.ModuleType("parent_identity_control")
exec("OPTIONS={'values':[1,2]}\ndef read(value=(1,2)):\n return OPTIONS,value\n",module.__dict__)
for method in (producer.canonical_module_sha256, lambda m:producer.canonical_module_slice_sha256(m,"read")):
 before=method(module)
 module.OPTIONS["values"].append(3)
 after=method(module)
 assert before!=after
 module.OPTIONS["values"].pop()
 assert method(module)==before
 out.extend([before,after])
print(json.dumps({"schema":"r7-parent-identity-parity-probe-v1","seed":20260925,"mutable_graphs":48,"module_controls":2,"digests_sha256":hashlib.sha256(json.dumps(out,separators=(",",":")).encode()).hexdigest(),"digests_count":len(out),"provider_calls":0},sort_keys=True))
