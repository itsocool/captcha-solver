// DOM·저장소 경계만 대체하고 실제 페이지 스크립트의 신뢰도 로직을 실행한다.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
class Element {
 constructor(value = '') { this.value = value; this.dataset = {}; this.customError = ''; this.textContent = ''; this.classes = new Set(); this.classList = {add: (...cs) => cs.forEach(c => this.classes.add(c)), remove: (...cs) => cs.forEach(c => this.classes.delete(c)), toggle: (c,on) => on ? this.classes.add(c) : this.classes.delete(c)}; }
 get valueAsNumber() { return this.value === '' ? NaN : Number(this.value); }
 setCustomValidity(value) { this.customError = value; }
 checkValidity() { return !this.customError && Number.isFinite(this.valueAsNumber) && this.valueAsNumber >= 0 && this.valueAsNumber <= 1; }
 addEventListener() {}
 replaceChildren() {}
}
const html = fs.readFileSync('apps/web/templates/data_source.html', 'utf8');
const defaults = id => html.match(new RegExp(`id="${id}"[^>]*value="([^"]+)"`))[1];
const high = new Element(defaults('autolabel-high-confidence'));
const low = new Element(defaults('autolabel-min-confidence'));
assert.equal(high.value, '0.92'); assert.equal(low.value, '0.86');
const elements = new Map([['#autolabel-high-confidence',high],['#autolabel-min-confidence',low]]);
const targets = new Element();targets.textContent='[]';elements.set('#data-source-targets', targets);
const storage = new Map();
const context = vm.createContext({
 document: {documentElement: {dataset: {}}, querySelector: selector => {if(!elements.has(selector)) elements.set(selector,new Element());return elements.get(selector);}},
 sessionStorage: {getItem:key=>storage.get(key)??null,setItem:(key,value)=>storage.set(key,value),removeItem:key=>storage.delete(key)},
 fetch: () => new Promise(()=>{}), URLSearchParams,
});
vm.runInContext(fs.readFileSync('apps/web/static/js/data_source.js','utf8'),context);
const run = code => vm.runInContext(code,context);
const label = new Element(), badge = new Element();
context.figure = {dataset: {},querySelector: selector => selector === 'input' ? label : badge};
for (const [score,level,display] of [[1,'high','1.00'],[0.92,'high','0.92'],[0.919,'medium','0.92'],[0.86,'medium','0.86'],[0.859,'low','0.86'],[0,'low','0.00']]) {
 run(`renderConfidence(figure, ${score})`);
 assert.equal(context.figure.dataset.confidenceLevel,level);
 assert.equal(badge.textContent,`신뢰도 ${display}`);
 assert.deepEqual([...label.classes],[...badge.classes]);
}
for (const score of ['undefined','null','NaN','Infinity','-0.1','1.01','"0.92"']) {
 run(`renderConfidence(figure, ${score})`);
 assert.equal(context.figure.dataset.confidenceLevel,'unknown');
 assert.equal(badge.textContent,'신뢰도 —');
}
high.value='0.95';run('renderConfidence(figure, 0.92)');assert.equal(context.figure.dataset.confidenceLevel,'medium');
for(const [h,l] of [['0.8','0.86'],['1.1','0.86'],['0.92','-0.1'],['','0.86']]) {
 high.value=h;low.value=l;assert.equal(run('readConfidenceThresholds()'),null);
}
high.value='0.92';low.value='0.86';assert.equal(run('readConfidenceThresholds().high'),0.92);
run('rememberConfidence("example.png", 0.92)');
assert.equal(run('readConfidence("example.png")'),0.92);
run('confidenceCache.clear()');assert.equal(run('readConfidence("example.png")'),undefined);
run('rememberConfidence("example.png", 0.92)');
elements.get('#captcha').value='other';assert.equal(run('readConfidence("example.png")'),undefined);
elements.get('#captcha').value='';elements.get('#rev').value='2';assert.equal(run('readConfidence("example.png")'),undefined);
elements.get('#rev').value='';run('rememberConfidence("example.png", undefined)');assert.equal(run('readConfidence("example.png")'),undefined);
console.log('신뢰도 기본값·경계·표시·입력 검증·저장·대상 분리 통과');
const gallery = elements.get('#gallery');
const makeCard = (confidence,createdOrder) => ({dataset:{confidence,createdOrder}});
let cards = [makeCard('',0),makeCard('0.92',1),makeCard('0.86',3),makeCard('0.86',2),makeCard('',4),makeCard('0',5)];
gallery.contains = () => false;
gallery.querySelectorAll = () => cards;
gallery.append = (...sorted) => {cards=sorted};
run('sortGallery()');assert.deepEqual(cards.map(c=>c.dataset.createdOrder),[1,2,3,5,0,4]);
gallery.contains = () => true;
cards.reverse();const editingOrder=[...cards];run('sortGallery()');assert.deepEqual(cards,editingOrder);
console.log('신뢰도 우선·생성순 동률·미예측 후순위·편집 중 정렬 보류 통과');
const smallFont = run('calculateLabelFontSize({width:160,height:60}, {width:80,height:56,labelWidth:80}, "123456")');
const largeFont = run('calculateLabelFontSize({width:160,height:60}, {width:260,height:56,labelWidth:260}, "123456")');
assert.ok(smallFont < largeFont);
assert.ok(smallFont >= 12 && largeFont <= 36);
assert.ok(run('calculateLabelFontSize({width:160,height:60}, {width:260,height:56,labelWidth:100}, "123456789012")') < largeFont);
assert.equal(run('calculateLabelFontSize({width:0,height:0}, {width:260,height:56,labelWidth:260}, "123456")'),22);
const editBadge = new Element();context.editFigure={querySelector:()=>editBadge};
run('renderEditBadge(editFigure, null)');assert.equal(editBadge.hidden,true);
run('renderEditBadge(editFigure, "2026-09-23T00:00:00.000Z")');assert.equal(editBadge.hidden,false);assert.ok(editBadge.title.includes('레이블 편집 저장됨'));
console.log('이미지 비율·입력 길이에 따른 글자 크기 및 편집 체크 표시 통과');

(async () => {
 // 저장 완료를 기다린 뒤 이동하고, 중복 요청과 실패 후 잠금을 검증한다.
 elements.get('#captcha').value='test';elements.get('#rev').value='1';
 elements.get('#gallery-refresh').disabled=false;
 elements.get('#move-to-train').disabled=false;
 gallery.querySelectorAll=()=>[];
 let reloads=0,calls=0,releaseSave;
 context.reloadForMove=async()=>{reloads++};
 run('loadGallery = reloadForMove');
 context.fetch=async()=>{calls++;return {ok:true,json:async()=>({moved:2,skipped:['111111.png'],failed:[]})}};
 context.pendingSave=new Promise(resolve=>{releaseSave=resolve});
 run('pendingLabelSaves.add(pendingSave)');
 const moving=run('moveCheckedToTrain()');
 assert.equal(elements.get('#move-to-train').disabled,true);
 assert.equal(calls,0);
 await run('moveCheckedToTrain()');assert.equal(calls,0);
 releaseSave();await moving;run('pendingLabelSaves.clear()');
 assert.equal(calls,1);assert.equal(reloads,1);
 assert.ok(elements.get('#move-to-train-status').textContent.includes('2장 이동'));
 assert.ok(elements.get('#move-to-train-status').textContent.includes('이름 중복 1장 유지'));
 assert.equal(elements.get('#move-to-train').disabled,false);
 context.fetch=async()=>({ok:false,status:409,json:async()=>({detail:'작업 중'})});
 await run('moveCheckedToTrain()');
 assert.equal(elements.get('#move-to-train-status').textContent,'이동 실패: 작업 중');
 assert.equal(run('movingToTrain'),false);assert.equal(reloads,2);
 console.log('체크 이미지 이동·저장 대기·중복 요청 차단·충돌 안내·오류 후 복구 통과');
})().catch(error=>{console.error(error);process.exitCode=1});
