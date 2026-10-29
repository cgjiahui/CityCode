// Cache only immutable, content-versioned scene resources in this origin.
// Failure or quota exhaustion falls back to ordinary HTTP without losing the scene.
export class TaskPool {
  constructor(limit){this.limit=limit;this.active=0;this.queue=[];}
  run(task,priority=0){return new Promise((resolve,reject)=>{this.queue.push({task,priority,resolve,reject});this.queue.sort((a,b)=>b.priority-a.priority);this.pump();});}
  pump(){while(this.active<this.limit&&this.queue.length){const job=this.queue.shift();this.active++;Promise.resolve().then(job.task).then(job.resolve,job.reject).finally(()=>{this.active--;this.pump();});}}
}

export class AssetStore {
  constructor(onProgress){
    this.onProgress=onProgress;this.pool=new TaskPool(6);this.inflight=new Map();this.writes=new Set();this.lastProgress=0;
    this.stats={networkBytes:0,cacheBytes:0,cacheHits:0,requests:0,coreBytes:0};
    try{this.cache=globalThis.caches?.open('citycode-content-v1').catch(()=>null)||Promise.resolve(null);}catch{this.cache=Promise.resolve(null);}
  }
  notify(force=false){const t=performance.now();if(force||t-this.lastProgress>250){this.lastProgress=t;this.onProgress?.(this.stats);}}
  get(url,{core=false,bytes,priority=0}={}){
    if(this.inflight.has(url))return this.inflight.get(url);
    const pending=this.pool.run(async()=>{
      const cache=await this.cache;let response;
      try{response=await cache?.match(url);}catch{}
      if(response){const data=await response.arrayBuffer();if(!bytes||data.byteLength===bytes){this.stats.cacheHits++;this.stats.cacheBytes+=data.byteLength;if(core)this.stats.coreBytes+=data.byteLength;this.notify(true);return data;}await cache.delete(url);}
      response=await fetch(url);if(!response.ok)throw Error(`Unable to load a resource（${response.status}）`);
      this.stats.requests++;
      const clone=cache?response.clone():null,reader=response.body.getReader(),chunks=[];let length=0;
      for(;;){const {done,value}=await reader.read();if(done)break;chunks.push(value);length+=value.byteLength;this.stats.networkBytes+=value.byteLength;if(core)this.stats.coreBytes+=value.byteLength;this.notify();}
      if(bytes&&length!==bytes)throw Error('A scene download is incomplete. Reload to try again.');
      const data=new Uint8Array(length);let offset=0;for(const chunk of chunks){data.set(chunk,offset);offset+=chunk.byteLength;}
      if(cache){const write=cache.put(url,clone).catch(()=>{}).finally(()=>this.writes.delete(write));this.writes.add(write);}
      this.notify(true);return data.buffer;
    },priority).finally(()=>this.inflight.delete(url));
    this.inflight.set(url,pending);return pending;
  }
}
