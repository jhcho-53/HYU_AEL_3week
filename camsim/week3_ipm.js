'use strict';
// Hand IPM inside a Colab output cell (inlined by camsim/week3_lab.py LaneClicker, which defines CONFIG and call()).
// call('page',[]) -> {image, points: null or 4 x [u,v] full-res px}
// call('save',[points]) -> {pitch_deg, height_m, lane_heading_deg}; the server checks the points (camsim.real.pose_from_lane_points).
// Points 1-2 lie on one tape, 3-4 on the other. The page draws each tape line and where the two meet (vanishing point).
(()=>{
const $=id=>document.getElementById(id), canvas=$('ipm'), ctx=canvas.getContext('2d');
canvas.width=CONFIG.width;canvas.height=CONFIG.height;
const S=CONFIG.scale, COLOR=['#ff4d4d','#3aa0ff'], GRAB=14;
const image=new Image();
let pts=[], drag=-1, busy=false, saved=false;
const msg=text=>{$('ipm-msg').textContent=text};

const line=(a,b)=>[a[1]-b[1],b[0]-a[0],a[0]*b[1]-a[1]*b[0]];        // the line through a and b, homogeneous
function vanishing(){
  if(pts.length<4)return null;
  const l=line(pts[0],pts[1]),m=line(pts[2],pts[3]),w=l[0]*m[1]-l[1]*m[0];
  return Math.abs(w)<1e-9?null:[(l[1]*m[2]-l[2]*m[1])/w,(l[2]*m[0]-l[0]*m[2])/w];
}

function guide(){
  const n=pts.length;
  if(saved)return;
  if(n===0)msg('1) 한쪽 테이프의 가운데를 클릭 (가까운 곳)');
  else if(n===1)msg('2) 같은 테이프의 가운데를 한 번 더 클릭 (먼 곳)');
  else if(n===2)msg('3) 다른 쪽 테이프의 가운데를 클릭 (가까운 곳)');
  else if(n===3)msg('4) 그 테이프의 가운데를 한 번 더 클릭 (먼 곳)');
  else msg('두 선이 테이프 위에 잘 겹치면 저장 (Enter). 점은 끌어서 옮길 수 있음');
}

function state(){$('ipm-state').textContent=saved?'저장함':pts.length<4?`점 ${pts.length} / 4`:'저장 안 함'}

function draw(){
  ctx.fillStyle='#303746';ctx.fillRect(0,0,canvas.width,canvas.height);
  if(image.complete&&image.naturalWidth)ctx.drawImage(image,0,0,canvas.width,canvas.height);
  const vp=vanishing();
  if(vp&&vp[1]>=0&&vp[1]<=canvas.height){                           // horizon: the row of the vanishing point
    ctx.strokeStyle='#ffffffb0';ctx.lineWidth=1;ctx.setLineDash([8,6]);
    ctx.beginPath();ctx.moveTo(0,vp[1]);ctx.lineTo(canvas.width,vp[1]);ctx.stroke();ctx.setLineDash([]);
    ctx.fillStyle='#fff';ctx.font='14px sans-serif';ctx.fillText('지평선 (소실점 높이)',6,vp[1]-6);
  }
  for(const k of [0,1]){
    const a=pts[2*k],b=pts[2*k+1];if(!b)continue;
    const d=[b[0]-a[0],b[1]-a[1]],far=vp&&vp[1]<Math.min(a[1],b[1])?vp:null;
    const t=d[1]?(canvas.height-a[1])/d[1]:0, near=[a[0]+d[0]*t,a[1]+d[1]*t];   // where the tape line leaves the bottom
    const up=d[1]<0?d:[-d[0],-d[1]], end=far||[a[0]+up[0]*20,a[1]+up[1]*20];    // on to the vanishing point, or far up
    ctx.strokeStyle=COLOR[k];ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(...(d[1]?near:a));ctx.lineTo(...end);ctx.stroke();
  }
  if(vp){ctx.fillStyle='#ffd400';ctx.beginPath();ctx.arc(vp[0],vp[1],6,0,Math.PI*2);ctx.fill()}
  pts.forEach((p,i)=>{
    ctx.fillStyle=COLOR[i>>1];ctx.strokeStyle='#fff';ctx.lineWidth=2;
    ctx.beginPath();ctx.arc(p[0],p[1],7,0,Math.PI*2);ctx.fill();ctx.stroke();
    ctx.fillStyle='#fff';ctx.font='bold 14px sans-serif';ctx.fillText(String(i+1),p[0]+9,p[1]-9);
  });
}

function changed(){saved=false;state();guide();draw()}

function at(e){const r=canvas.getBoundingClientRect();
  return [(e.clientX-r.left)*canvas.width/r.width,(e.clientY-r.top)*canvas.height/r.height]}

canvas.addEventListener('pointerdown',e=>{
  if(busy)return;
  const q=at(e),i=pts.findIndex(p=>Math.hypot(p[0]-q[0],p[1]-q[1])<=GRAB);
  if(i>=0){drag=i;canvas.setPointerCapture(e.pointerId);return}
  if(pts.length>=4){msg('점 4개를 다 찍었음. 옮기려면 점을 끌고, 새로 하려면 "처음부터 다시"');return}
  pts.push(q);drag=pts.length-1;canvas.setPointerCapture(e.pointerId);changed();
});
canvas.addEventListener('pointermove',e=>{if(drag>=0&&!busy){pts[drag]=at(e);changed()}});
canvas.addEventListener('pointerup',()=>{drag=-1});

async function save(){
  if(busy)return;
  if(pts.length<4){msg(`점을 4개 찍어야 저장할 수 있음 (지금 ${pts.length}개)`);return}
  busy=true;
  try{const r=await call('save',[pts.map(p=>[p[0]/S,p[1]/S])]);saved=true;state();
    const height=r.height_m==null?'높이는 3-3 에서 계산':`높이 ${(r.height_m*100).toFixed(1)} cm`;
    msg(`저장함: 카메라 기울기 ${r.pitch_deg>=0?'+':''}${r.pitch_deg.toFixed(2)}도, ${height}. 이제 3-3 셀부터 끝까지 다시 실행할 것`)}
  catch(e){msg(e.message)}finally{busy=false}
}
const undo=()=>{if(!busy&&pts.length){pts.pop();changed()}};
const reset=()=>{if(!busy){pts=[];changed()}};
document.addEventListener('keydown',e=>{
  if(e.key==='Enter'){e.preventDefault();save()}else if(e.key==='Backspace'){e.preventDefault();undo()}});
$('ipm-save').onclick=save;$('ipm-undo').onclick=undo;$('ipm-reset').onclick=reset;
image.onload=draw;

(async()=>{
  busy=true;msg('불러오는 중. 모두 실행 중이면 다른 셀이 다 끝난 뒤에 뜸');
  try{const r=await call('page',[]);image.src=r.image;
    if(r.points){pts=r.points.map(p=>[p[0]*S,p[1]*S]);saved=true;msg('저장된 점을 불러왔음. 고치려면 점을 끌고 다시 저장 (Enter)')}
    state();guide();draw()}
  catch(e){msg(e.message)}finally{busy=false}
})();
})();
