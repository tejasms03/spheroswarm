const fs=require('fs');const Q=require('./ptr_real.js');
const AW=138.8,AH=110.8;
const brg=(a,b)=>((Math.atan2(b[0]-a[0],b[1]-a[1])*180/Math.PI)+360)%360;
const r1=v=>Math.round(v*10)/10;const P=p=>[r1(p[0]),r1(p[1])];
function rngOf(seed){return function(){seed|=0;seed=seed+0x6D2B79F5|0;let t=Math.imul(seed^seed>>>15,1|seed);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296}}
const scen=[];
const desc={cross:'A drives left to right while B drives top to bottom through the same crossing.',
 head:'A and B drive at each other along the same line.',
 orbit:'A orbits the middle for 60 s; B is parked on the circle and must be moved out of the way.',
 three:'Three balls cross the middle from three sides.',
 corridor:'A and B patrol the same horizontal line in opposite directions, 2 rounds each.',
 orbits:'A and B orbit two circles that touch in the middle, in opposite directions, 60 s.'};
for(const [name,list] of Object.entries(Q.CASES)){
  const balls=[],jobs=[];
  list.forEach((c,i)=>{const slot=String.fromCharCode(65+i);const[kind,p,arg]=c;let face=0;
    if(kind==='goto'){jobs.push({ball:slot,kind:'p2p',to:P(arg[0])});face=brg(p,arg[0])}
    if(kind==='orbit'){jobs.push({ball:slot,kind:'orbit',centre:P(arg.c),radius:arg.r,direction:arg.dir===1?'cw':'ccw',seconds:arg.dur});face=0}
    if(kind==='patrol'){jobs.push({ball:slot,kind:'patrol',points:arg.map(P),rounds:2});face=brg(p,arg[0])}
    balls.push({slot,start:P(p),facing_deg:Math.round(face),parked:kind==='park'})});
  scen.push({file:`base_${name}.json`,name:`base: ${name}`,description:desc[name],arena_cm:[AW,AH],duration_s:120,balls,jobs})}
function stress(seed,n,dur){const R=rngOf(seed),M=20,rp=()=>P([M+R()*(AW-2*M),M+R()*(AH-2*M)]);
  const balls=[],jobs=[];
  for(let i=0;i<n;i++){let p,ok=false;for(let k=0;k<300&&!ok;k++){p=rp();ok=balls.every(b=>Math.hypot(b.start[0]-p[0],b.start[1]-p[1])>35)}
    balls.push({slot:String.fromCharCode(65+i),start:p,facing_deg:Math.round(R()*360)})}
  for(const b of balls)for(let k=0;k<10;k++){const x=R();
    if(x<0.3)jobs.push({ball:b.slot,kind:'p2p',to:rp()});
    else if(x<0.45)jobs.push({ball:b.slot,kind:'line',from:rp(),to:rp()});
    else if(x<0.6)jobs.push({ball:b.slot,kind:'poly',points:[rp(),rp(),rp()]});
    else if(x<0.75){const r=25+Math.round(R()*6);jobs.push({ball:b.slot,kind:'orbit',centre:P([r+M+R()*(AW-2*(r+M)),r+M+R()*(AH-2*(r+M))]),radius:r,direction:R()<0.5?'cw':'ccw',seconds:20+Math.round(R()*10)})}
    else if(x<0.9)jobs.push({ball:b.slot,kind:'patrol',points:[rp(),rp(),rp()],rounds:1});
    else jobs.push({ball:b.slot,kind:'park',seconds:5+Math.round(R()*5)})}
  return {file:`stress_${n}balls_seed${seed}.json`,name:`stress: ${n} balls, seed ${seed}`,description:`Each ball works through its own list of 10 jobs, one after another with a 3 s pause, until ${dur} s. Same jobs every run.`,arena_cm:[AW,AH],duration_s:dur,balls,jobs}}
scen.push(stress(11,2,180),stress(12,2,180),stress(21,3,180));
fs.writeFileSync('scenarios_raw.json',JSON.stringify(scen));
console.log(scen.map(s=>s.file+' jobs '+s.jobs.length).join('\n'));
