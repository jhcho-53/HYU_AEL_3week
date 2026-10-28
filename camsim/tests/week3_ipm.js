'use strict';
// camsim/week3_ipm.js in a node vm with just enough DOM for its logic (run by test_week3_lab.py).
// node week3_ipm.js <scenario>: exits non-zero on a failed check.
const assert=require('assert'),fs=require('fs'),path=require('path'),vm=require('vm');
const JS=fs.readFileSync(path.join(__dirname,'..','week3_ipm.js'),'utf8');
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
const plain=x=>JSON.parse(JSON.stringify(x));      // arrays made inside the vm have its own Array prototype
const SCALE=.5, W=960, H=600;

class El{
  constructor(tag){Object.assign(this,{tag,width:0,height:0,text:'',on:{}})}
  get textContent(){return this.text}
  set textContent(v){this.text=String(v)}
  addEventListener(type,f){(this.on[type]=this.on[type]||[]).push(f)}
  fire(type,props){const e=Object.assign({type,preventDefault(){}},props);for(const f of this.on[type]||[])f(e)}
  getBoundingClientRect(){return {left:0,top:0,width:this.width,height:this.height}}   // 1 screen px = 1 canvas px
  setPointerCapture(){}
  getContext(){return new Proxy({},{get:(o,k)=>k in o?o[k]:()=>{},set:(o,k,v)=>(o[k]=v,true)})}
}

// What camsim.week3_lab.LaneClicker answers; the real one also checks the points with real.pose_from_lane_points.
async function page({points=null,fail=false,height=.1234}={}){
  const els={},document=new El('document');
  document.getElementById=id=>els[id]||(els[id]=new El(id));
  const calls=[];
  const server={page:()=>({image:'data:image/jpeg;base64,x',points}),
    save:pts=>{if(fail)throw Error('두 선이 영상에서 평행이라 만나지 않음. 테스트');return {pitch_deg:-1.25,height_m:height,lane_heading_deg:0}}};
  const call=async(name,args)=>{calls.push([name,...args]);return server[name](...args)};
  const CONFIG={width:W,height:H,scale:SCALE,prefix:'t.'};
  const context=vm.createContext({document,CONFIG,call,setTimeout,
    Image:class{set src(v){this._src=v;this.complete=true;this.naturalWidth=W;setTimeout(()=>this.onload&&this.onload())}get src(){return this._src}}});
  vm.runInContext(JS,context,{filename:'week3_ipm.js'});
  const p={els,calls,text:id=>els[id].textContent,settle:()=>sleep(20)};
  const fire=(type,u,v)=>els.ipm.fire(type,{clientX:u,clientY:v,pointerId:1});
  p.click=async(u,v)=>{fire('pointerdown',u,v);fire('pointerup',u,v);await p.settle()};
  p.drag=async(from,to)=>{fire('pointerdown',...from);fire('pointermove',...to);fire('pointerup',...to);await p.settle()};
  p.key=async key=>{document.fire('keydown',{key});await p.settle()};
  p.saves=()=>calls.filter(c=>c[0]==='save');
  p.four=async()=>{for(const q of [[300,560],[420,330],[700,560],[560,330]])await p.click(...q)};
  await p.settle();
  return p;
}

const scenarios={
  async loadsAndGuides(){
    const p=await page();
    assert.deepStrictEqual(p.calls,[['page']]);
    assert.strictEqual(p.els.ipm.width,W);assert.strictEqual(p.els.ipm.height,H);
    assert.match(p.text('ipm-msg'),/^1\) 한쪽 테이프의 가운데를 클릭/);
    assert.strictEqual(p.text('ipm-state'),'점 0 / 4');
  },
  async guidesEachOfTheFourPoints(){
    const p=await page();
    await p.click(300,560);assert.match(p.text('ipm-msg'),/^2\) 같은 테이프/);
    await p.click(420,330);assert.match(p.text('ipm-msg'),/^3\) 다른 쪽 테이프/);
    await p.click(700,560);assert.match(p.text('ipm-msg'),/^4\) /);
    await p.click(560,330);assert.match(p.text('ipm-msg'),/저장 \(Enter\)/);
    assert.strictEqual(p.text('ipm-state'),'저장 안 함');
  },
  async enterSavesFullResolutionPoints(){
    const p=await page();
    await p.four();await p.key('Enter');
    assert.deepStrictEqual(plain(p.saves()),[['save',[[600,1120],[840,660],[1400,1120],[1120,660]]]]);
    assert.strictEqual(p.text('ipm-msg'),'저장함: 카메라 기울기 -1.25도, 높이 12.3 cm. 이제 3-3 셀부터 끝까지 다시 실행할 것');
    assert.strictEqual(p.text('ipm-state'),'저장함');
  },
  async heightIsLeftToTheNextCellWithoutALaneWidth(){
    const p=await page({height:null});
    await p.four();await p.key('Enter');
    assert.strictEqual(p.text('ipm-msg'),'저장함: 카메라 기울기 -1.25도, 높이는 3-3 에서 계산. 이제 3-3 셀부터 끝까지 다시 실행할 것');
  },
  async enterBeforeFourPointsWarns(){
    const p=await page();
    await p.click(300,560);await p.click(420,330);await p.key('Enter');
    assert.strictEqual(p.saves().length,0);
    assert.match(p.text('ipm-msg'),/점을 4개 찍어야 저장할 수 있음 \(지금 2개\)/);
  },
  async fifthClickWarnsAndDragMovesAPoint(){
    const p=await page();
    await p.four();await p.key('Enter');
    await p.click(100,100);
    assert.match(p.text('ipm-msg'),/다 찍었음/);
    await p.drag([420,330],[430,340]);                         // grab point 2 and move it
    assert.strictEqual(p.text('ipm-state'),'저장 안 함');
    await p.key('Enter');
    assert.deepStrictEqual(plain(p.saves()[1][1][1]),[860,680]);
  },
  async backspaceAndResetRemovePoints(){
    const p=await page();
    await p.four();await p.key('Backspace');
    assert.strictEqual(p.text('ipm-state'),'점 3 / 4');
    p.els['ipm-reset'].onclick();await p.settle();
    assert.strictEqual(p.text('ipm-state'),'점 0 / 4');
    assert.match(p.text('ipm-msg'),/^1\) /);
  },
  async savedPointsComeBack(){
    const p=await page({points:[[600,1120],[840,660],[1400,1120],[1120,660]]});
    assert.strictEqual(p.text('ipm-state'),'저장함');
    assert.match(p.text('ipm-msg'),/저장된 점을 불러왔음/);
    await p.key('Enter');
    assert.deepStrictEqual(plain(p.saves()[0][1]),[[600,1120],[840,660],[1400,1120],[1120,660]]);
  },
  async serverErrorIsShown(){
    const p=await page({fail:true});
    await p.four();await p.key('Enter');
    assert.match(p.text('ipm-msg'),/만나지 않음\. 테스트/);
    assert.strictEqual(p.text('ipm-state'),'저장 안 함');
  },
};

const name=process.argv[2];
if(!scenarios[name]){console.error('unknown scenario '+name);process.exit(2)}
scenarios[name]().then(()=>process.exit(0),e=>{console.error(e);process.exit(1)});
