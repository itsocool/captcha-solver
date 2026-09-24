const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const code = fs.readFileSync('apps/web/static/js/data_source.js', 'utf8');
const start = code.indexOf('function openTaskStream(');
assert.ok(start >= 0, 'HTTP 오류 본문을 읽는 공용 SSE 클라이언트 필요');
const context = vm.createContext({AbortController, TextDecoder, fetch: null});
vm.runInContext(code.slice(start, code.indexOf('\nfunction finish(', start)), context);
const tick = () => new Promise(resolve => setImmediate(resolve));
async function collect(response, closeOnSummary=true) {
 let calls=0;
 context.fetch = async () => {calls++;return response;};
 const stream = context.openTaskStream('/api/v1/data-source/stream');
 const events=[];
 for(const name of ['start','item','summary','error']) stream.addEventListener(name, e=>{
  events.push({name,data:JSON.parse(e.data)});
  if(name==='error'||(name==='summary'&&closeOnSummary)) stream.close();
 });
 for(let i=0;i<12;i++) await tick();
 assert.equal(calls,1, '자동 재접속으로 수집을 중복 실행하지 않는다');
 return events;
}
(async()=>{
 let events=await collect(new Response(JSON.stringify({detail:'이미 다른 수집이 실행 중입니다'}),{status:409}));
 assert.equal(events[0].data.message,'이미 다른 수집이 실행 중입니다');
 events=await collect(new Response(JSON.stringify({detail:[{msg:'파일 개수 오류'}]}),{status:422}));
 assert.equal(events[0].data.message,'파일 개수 오류');
 const payload='event: start\r\ndata: {"total":1}\r\n\r\nevent: item\ndata: {"name":"한글.png"}\n\nevent: summary\ndata: {"saved":1}\n\n';
 const bytes=new TextEncoder().encode(payload);
 const body=new ReadableStream({start(c){for(const byte of bytes)c.enqueue(new Uint8Array([byte]));c.close();}});
 events=await collect(new Response(body,{headers:{'Content-Type':'text/event-stream'}}));
 assert.deepEqual(events.map(e=>e.name),['start','item','summary']);
 assert.equal(events[1].data.name,'한글.png');
 events=await collect(new Response('event: start\ndata: {}\n\n',{headers:{'Content-Type':'text/event-stream'}}));
 assert.equal(events.at(-1).name,'error');
 assert.match(events.at(-1).data.message,/연결/);
 events=await collect(new Response('<html>proxy error</html>'));
 assert.equal(events[0].name,'error');
 let aborted=false;
 context.fetch=(_url,{signal})=>new Promise((_resolve,reject)=>signal.addEventListener('abort',()=>{aborted=true;reject(new DOMException('Aborted','AbortError'));}));
 const stream=context.openTaskStream('/stream');
 stream.addEventListener('error',()=>assert.fail('사용자 중단을 오류로 표시하지 않는다'));
 stream.close();await tick();assert.equal(aborted,true);
 console.log('HTTP 409/422·UTF-8/분할 프레임·완료·전송 중단·사용자 취소·재접속 방지 통과');
})().catch(e=>{console.error(e);process.exitCode=1;});
