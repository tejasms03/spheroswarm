const AW=138.8,AH=110.8,TURN=120,SLEW=+((typeof process!=="undefined"&&process.env.SLEW)||90),BR=3.65,DT=1/30;
const M=12,LOOK=12,E=typeof process!=='undefined'?process.env:{};
let S=+(E.S||12.1),CMDLAT=+(E.CMDLAT||0.37),CAMLAT=+(E.CAMLAT||0.1),PNOISE=+(E.PNOISE||1.0),FNOISE=+(E.FNOISE||5),DROP=+(E.DROP||0.012),
  SPDVAR=+(E.SPDVAR||0.25),TAU_UP=+(E.TAU_UP||0.35),TAU_DN=+(E.TAU_DN||0.5),SPINLO=+(E.SPINLO||40),SPINHI=+(E.SPINHI||140),RUNON=+(E.RUNON||23),GAIN=+(E.GAIN||1.2),BIASW=+(E.BIASW===undefined?6:E.BIASW),BIASMAX=+(E.BIASMAX||15),REAL=+(E.REAL===undefined?1:E.REAL);
let HOR=+(E.HOR||6),PDT=+(E.PDT||0.25),REPLAN=+(E.REPLAN||1),SEP=+(E.SEP||20),TMARG=+(E.TMARG||0.5),COMMIT=+(E.COMMIT||1),PRED=+(E.PRED||16),AGE=+(E.AGE||2),STARVE=+(E.STARVE||4);
function rngOf(seed){return function(){seed|=0;seed=seed+0x6D2B79F5|0;let t=Math.imul(seed^seed>>>15,1|seed);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296}}
const dirOf=d=>{const r=d*Math.PI/180;return[Math.sin(r),Math.cos(r)]};
const brg=(dx,dy)=>((Math.atan2(dx,dy)*180/Math.PI)+360)%360;
const wrap=a=>((a%360)+540)%360-180;
const sub=(a,b)=>[a[0]-b[0],a[1]-b[1]],add=(a,b)=>[a[0]+b[0],a[1]+b[1]],mulv=(a,k)=>[a[0]*k,a[1]*k];
const dot=(a,b)=>a[0]*b[0]+a[1]*b[1],len=a=>Math.hypot(a[0],a[1]);
const clamp=(v,lo,hi)=>Math.max(lo,Math.min(hi,v));
function mkPath(pts,step=2){const out=[pts[0].slice()];for(let a=0;a<pts.length-1;a++){const d=sub(pts[a+1],pts[a]),n=Math.max(1,Math.ceil(len(d)/step));for(let k=1;k<=n;k++)out.push(add(pts[a],mulv(d,k/n)))}
  const cum=[0];for(let k=1;k<out.length;k++)cum.push(cum[k-1]+len(sub(out[k],out[k-1])));return{pts:out,cum,total:cum[cum.length-1]}}
function at(P,s){s=clamp(s,0,P.total);let lo=0,hi=P.cum.length-1;while(hi-lo>1){const m=(lo+hi)>>1;if(P.cum[m]<=s)lo=m;else hi=m}
  const span=P.cum[hi]-P.cum[lo],f=span<1e-9?0:(s-P.cum[lo])/span;return add(P.pts[lo],mulv(sub(P.pts[hi],P.pts[lo]),f))}
function project(P,p,s0){let best=s0,bd=1e9;const lo=Math.max(0,s0-5),hi=Math.min(P.total,s0+30);
  const cand=[];for(let k=0;k<P.pts.length;k++){if(P.cum[k]<lo||P.cum[k]>hi)continue;const d=len(sub(P.pts[k],p));cand.push([P.cum[k],d]);if(d<bd){bd=d;best=P.cum[k]}}
  return[best,bd]}
function chaikin(pts,it=3){let p=pts;for(let r=0;r<it;r++){const q=[p[0]];for(let k=0;k<p.length-1;k++){const a=p[k],c=p[k+1];q.push(add(mulv(a,0.75),mulv(c,0.25)),add(mulv(a,0.25),mulv(c,0.75)))}q.push(p[p.length-1]);p=q}return p}
function World(n,seed,scen){
  const W={t:0,rng:rngOf(seed),balls:[],stats:{minD:1e9,contact:0,near:0,brake:0,frames:0,jobsDone:0,jobsStarted:0,stuck:0,asides:0,detours:0},log:[],seq:0,lastPlan:-1e9,book:[]};
  const R=W.rng,rp=()=>[M+R()*(AW-2*M),M+R()*(AH-2*M)];W.randPoint=rp;
  for(let i=0;i<n;i++){let p,ok=false;for(let k=0;k<200&&!ok;k++){p=rp();ok=W.balls.every(b=>len(sub(b.p,p))>25)}
    W.balls.push({name:String.fromCharCode(65+i),i,p,f:R()*360,v:[0,0],spd:0,job:{kind:'idle',until:R()*3},rank:1e9,state:'idle',path:null,s:0,knots:null,blockedBy:null,waitSince:null,lastMoveP:p.slice(),lastMoveT:0,wasContact:{},stuckFlag:false,brake:false})}
  if(scen&&typeof scen==='object')setupScript(W,scen);
  else if(scen&&scen!=='stress')setupCase(W,scen);
  return W}
const CASES={
  cross:[['goto',[20,55],[[120,55]]],['goto',[70,12],[[70,100]]]],
  head:[['goto',[20,55],[[120,55]]],['goto',[120,55],[[20,55]]]],
  orbit:[['orbit',[70,25],{c:[70,55],r:30,dir:1,dur:60}],['park',[70,85]]],
  three:[['goto',[25,20],[[115,90]]],['goto',[115,20],[[25,90]]],['goto',[70,100],[[70,12]]]],
  corridor:[['patrol',[25,55],[[115,55],[25,55]]],['patrol',[115,55],[[25,55],[115,55]]]],
  orbits:[['orbit',[25,55],{c:[50,55],r:25,dir:1,dur:60}],['orbit',[115,55],{c:[90,55],r:25,dir:-1,dur:60}]]};
function setupCase(W,scen){
  W.fixed=true;W.balls=[];
  CASES[scen].forEach((c,i)=>{const[kind,p,arg]=c;
    const b={name:String.fromCharCode(65+i),i,p:p.slice(),f:0,v:[0,0],spd:0,job:{kind:'idle',until:Infinity},rank:1e9,state:'idle',path:null,s:0,knots:null,blockedBy:null,waitSince:null,lastMoveP:p.slice(),lastMoveT:0,wasContact:{},stuckFlag:false,brake:false};
    W.balls.push(b);
    if(kind==='park')return;
    let job;if(kind==='orbit')job={kind:'orbit',c:arg.c,r:arg.r,dir:arg.dir,until:arg.dur};else if(kind==='patrol')job={kind:'patrol',pts:arg,rounds:2};else job={kind,pts:arg};
    b.job=job;b.jobT=0;b.state='go';b.rank=++W.seq;W.stats.jobsStarted++;b.path=jobPath(W,b,job);b.s=0;
    const q=at(b.path,Math.min(12,b.path.total));b.f=brg(q[0]-p[0],q[1]-p[1]);
    log(W,b.name+' starts '+kind)})}
function mkBall(i,p,f){return{name:String.fromCharCode(65+i),i,p:p.slice(),f:f||0,v:[0,0],spd:0,job:{kind:'idle',until:0},rank:1e9,state:'idle',path:null,s:0,knots:null,blockedBy:null,waitSince:null,lastMoveP:p.slice(),lastMoveT:0,wasContact:{},stuckFlag:false,brake:false}}
function setupScript(W,sc){
  W.script=true;W.balls=[];
  sc.balls.forEach((bb,i)=>{const b=mkBall(i,bb.start,bb.facing||0);b.queue=sc.jobs.filter(j=>j.ball===b.name).map(j=>Object.assign({},j));b.job={kind:'idle',until:0};W.balls.push(b)})}
function nextScripted(W,b){
  const j=b.queue.shift();if(!j){b.job={kind:'idle',until:Infinity};return}
  let job;
  if(j.kind==='p2p')job={kind:'goto',pts:[j.to]};
  else if(j.kind==='line')job={kind:'line',pts:[j.from,j.to]};
  else if(j.kind==='poly')job={kind:'poly',pts:j.points};
  else if(j.kind==='orbit')job={kind:'orbit',c:j.centre,r:j.radius,dir:j.direction==='cw'?1:-1,until:W.t+j.seconds};
  else if(j.kind==='patrol')job={kind:'patrol',pts:j.points,rounds:j.rounds||1};
  else {b.job={kind:'idle',until:W.t+(j.seconds||5)};b.state='idle';return}
  b.job=job;b.jobT=W.t;b.lastMoveP=b.p.slice();b.lastMoveT=W.t;b.knots=null;b.blockedBy=null;b.waitSince=null;b.state='go';b.rank=++W.seq;W.stats.jobsStarted++;b.path=jobPath(W,b,job);b.s=0;
  log(W,b.name+' starts '+j.kind)}
function log(W,s){W.log.push(W.t.toFixed(1).padStart(6)+'s  '+s);if(W.log.length>4000)W.log.shift()}
function jobPath(W,b,job){
  if(job.kind==='goto')return mkPath([b.p,job.pts[0]]);
  if(job.kind==='line'||job.kind==='poly')return mkPath(chaikin([b.p].concat(job.pts)));
  if(job.kind==='patrol'){let seq=[b.p];for(let r=0;r<job.rounds;r++)seq=seq.concat(job.pts);seq.push(job.pts[0]);return mkPath(chaikin(seq))}
  if(job.kind==='orbit'){const j=job;const a0=Math.atan2(b.p[1]-j.c[1],b.p[0]-j.c[0]);const laps=Math.ceil(S*(j.until-W.t)/(2*Math.PI*j.r))+1;const pts=[b.p];
    for(let k=0;k<=laps*72;k++){const a=a0+j.dir*k*Math.PI/36;pts.push([j.c[0]+j.r*Math.cos(a),j.c[1]+j.r*Math.sin(a)])}return mkPath(chaikin(pts.slice(0,8),3).concat(pts.slice(8)))}
  return null}
function newJob(W,b){
  const R=W.rng,rp=W.randPoint,x=R();let job;
  if(x<0.25)job={kind:'goto',pts:[rp()]};
  else if(x<0.4)job={kind:'line',pts:[rp(),rp()]};
  else if(x<0.55){const k=3+Math.floor(R()*3),pts=[];for(let i=0;i<k;i++)pts.push(rp());job={kind:'poly',pts}}
  else if(x<0.72){const r=18+R()*14;const c=[r+M+R()*(AW-2*(r+M)),r+M+R()*(AH-2*(r+M))];job={kind:'orbit',c,r,dir:R()<0.5?1:-1,until:W.t+20+R()*25}}
  else if(x<0.9){const k=3+Math.floor(R()*2),pts=[];for(let i=0;i<k;i++)pts.push(rp());job={kind:'patrol',pts,rounds:2}}
  else job={kind:'idle',until:W.t+4+R()*10};
  b.job=job;b.jobT=W.t;b.lastMoveP=b.p.slice();b.lastMoveT=W.t;b.knots=null;b.blockedBy=null;b.waitSince=null;
  if(job.kind==='idle'){b.state='idle';b.rank=1e9;b.path=null}else{b.state='go';b.rank=++W.seq;W.stats.jobsStarted++;b.path=jobPath(W,b,job);b.s=0}
  log(W,b.name+' new job: '+job.kind)}
function finish(W,b){W.stats.jobsDone++;log(W,b.name+' finished '+b.job.kind+' in '+(W.t-b.jobT).toFixed(0)+'s');b.state='done';b.rank=1e9;b.path=null;b.knots=null;b.job={kind:'idle',until:W.fixed?Infinity:(W.script?W.t:W.t+1+W.rng()*4)}}
const moving=b=>b.state==='go'||b.state==='aside';
function prio(W,b){if(b.state==='aside')return -1e6+(b.asideT||0);if(b.state!=='go')return 1e9;return b.rank-AGE*(b.waitSince===null?0:W.t-b.waitSince)}
function clash(W,id,p,t0,t1){for(const q of W.book){if(q.id===id)continue;if(q.t1+TMARG<t0||q.t0-TMARG>t1)continue;if(len(sub(q.p,p))<SEP)return q.id}return -1}
function planAll(W){
  W.book=[];const B=W.balls;
  for(const b of B){const coast=moving(b)?at(b.path,b.s+Math.min(6,b.spd*0.45)):b.p;
    W.book.push({id:b.i,p:b.p,t0:0,t1:0.75},{id:b.i,p:coast,t0:0,t1:0.75});
    if(!moving(b))W.book.push({id:b.i,p:b.p,t0:0,t1:HOR+1})}
  const order=B.filter(moving).sort((a,b)=>prio(W,a)-prio(W,b));
  for(const b of order)planOne(W,b)}
function planOne(W,b){
  const P=b.path;let t=0,s=b.s;const knots=[[0,s]];b.blockedBy=null;b.mustYield=null;let firstBlock=null;
  while(t<HOR-1e-9){
    const s2=Math.min(P.total,s+S*PDT),p2=at(P,s2);
    let who=clash(W,b.i,p2,t,t+PDT);
    if(who<0&&s2>s+0.01){const w2=clash(W,b.i,p2,t,t+COMMIT);if(w2>=0)who=w2}
    if(who<0||s2<=s+0.01){if(s2>s+0.01)W.book.push({id:b.i,p:p2,t0:t,t1:t+PDT});else W.book.push({id:b.i,p:p2,t0:t,t1:t+PDT});s=s2;t+=PDT;knots.push([t,s])}
    else{if(firstBlock===null)firstBlock=who;const stayWho=clash(W,b.i,at(P,s),t,t+PDT);if(stayWho>=0&&b.mustYield==null)b.mustYield=stayWho;
      W.book.push({id:b.i,p:at(P,s),t0:t,t1:t+PDT});t+=PDT;knots.push([t,s])}}
  W.book.push({id:b.i,p:at(P,s),t0:HOR,t1:HOR+1});
  b.knots=knots;b.planT=W.t;
  const progressing=knots[Math.min(4,knots.length-1)][1]>b.s+1;
  if(!progressing){b.blockedBy=firstBlock;if(b.waitSince===null)b.waitSince=W.t}else{b.waitSince=null}}
function allowed(b,t){const k=b.knots,x=t-b.planT;if(!k)return b.s;if(x<=0)return k[0][1];for(let i=1;i<k.length;i++){if(k[i][0]>=x){const f=(x-k[i-1][0])/(k[i][0]-k[i-1][0]);return k[i-1][1]+f*(k[i][1]-k[i-1][1])}}return k[k.length-1][1]}
function aside(W,lo,hi){
  const hp=hi.path?hi.path.pts.filter((q,k)=>hi.path.cum[k]>=hi.s-5&&hi.path.cum[k]<=hi.s+80):[hi.p];
  const others=W.balls.filter(o=>o!==lo&&o!==hi),otherPts=[].concat(...others.map(o=>o.path?o.path.pts.filter((q,k)=>o.path.cum[k]>=o.s&&o.path.cum[k]<=o.s+60):[o.p]));
  let best=null;
  for(const reach of[30,45])for(let k=0;k<8;k++){const a=k*Math.PI/4,g=[clamp(lo.p[0]+reach*Math.cos(a),8,AW-8),clamp(lo.p[1]+reach*Math.sin(a),8,AH-8)];
    if(len(sub(g,lo.p))<15)continue;const clear=Math.min(...hp.map(q=>len(sub(q,g))));const crowd=otherPts.length?Math.min(...otherPts.map(q=>len(sub(q,g)))):99;
    const sc=Math.min(clear,40)+0.6*Math.min(crowd,35);if(!best||sc>best.sc)best={g,sc}}
  if(!best)return;
  if(lo.state==='go'){const rest=lo.path.pts.filter((q,k)=>lo.path.cum[k]>=lo.s);lo.resumeJob={path:mkPath([best.g].concat(rest.length?rest:[lo.p])),kind:lo.job.kind}}
  else lo.resumeJob=null;
  lo.asideFor=hi.i;lo.asideHold=null;lo.prevState=lo.state;lo.state='aside';lo.path=mkPath([lo.p,best.g]);lo.s=0;lo.knots=null;lo.waitSince=null;lo.asideT=W.t;W.stats.asides++;
  log(W,lo.name+' moves aside for '+hi.name)}

function gauss(R){let u=0,v=0;while(u===0)u=R();while(v===0)v=R();return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v)}
function observe(W,b){
  if(!REAL)return{p:b.p,f:b.f,v:b.v,spd:b.spd};
  const h=b.hist||[];const tt=W.t-CAMLAT;let k=h.length-1;while(k>0&&h[k].t>tt)k--;const x=h[Math.max(0,k)]||{p:b.p,f:b.f,v:b.v,spd:b.spd};
  if(b.lastObs&&W.rng()<DROP)return b.lastObs;
  const o={p:[x.p[0]+PNOISE*gauss(W.rng),x.p[1]+PNOISE*gauss(W.rng)],f:(x.f+FNOISE*gauss(W.rng)+360)%360,v:x.v,spd:x.spd};b.lastObs=o;return o}
function swapIn(b,o){b._t={p:b.p,f:b.f,v:b.v,spd:b.spd};b.p=o.p;b.f=o.f;b.v=o.v;b.spd=o.spd}
function swapOut(b){const t=b._t;b.p=t.p;b.f=t.f;b.v=t.v;b.spd=t.spd}
function step(W){
  W.t+=DT;const B=W.balls;W.stats.frames+=B.length;
  for(const b of B){(b.hist=b.hist||[]).push({t:W.t,p:b.p.slice(),f:b.f,v:b.v.slice(),spd:b.spd});if(b.hist.length>30)b.hist.shift();
    if(b.bias===undefined)b.bias=0;if(REAL){b.bias=clamp(b.bias+BIASW*Math.sqrt(DT)*gauss(W.rng),-BIASMAX,BIASMAX)}
    if(b.spdF===undefined){b.spdF=1;b.spdT=0}if(REAL&&W.t-b.spdT>3){b.spdF=clamp(b.spdF+0.15*gauss(W.rng),1-SPDVAR,1+SPDVAR);b.spdT=W.t}}
  const obs=B.map(b=>observe(W,b));
  if(W.script){for(const b of B)if((b.state==='idle'||b.state==='done')&&W.t>=(b.job.until||0)&&b.spd<0.5){if(b.state==='done'&&b.pauseUntil===undefined){b.pauseUntil=W.t+3;continue}if(b.pauseUntil!==undefined&&W.t<b.pauseUntil)continue;b.pauseUntil=undefined;swapIn(b,obs[b.i]);nextScripted(W,b);swapOut(b)}}
  else if(!W.fixed)for(const b of B)if((b.state==='idle'||b.state==='done')&&W.t>=(b.job.until||0)&&b.spd<0.5){swapIn(b,obs[b.i]);newJob(W,b);swapOut(b)}
  if(W.t-W.lastPlan>=REPLAN){B.forEach(b=>swapIn(b,obs[b.i]));planAll(W);W.lastPlan=W.t;
    for(const b of B){if(b.state!=='aside')continue;
      if(b.knots&&b.knots[Math.min(4,b.knots.length-1)][1]<=b.s+1){if(b.asideHold==null)b.asideHold=W.t}else b.asideHold=null;
      if(b.asideHold!=null&&W.t-b.asideHold>6){log(W,b.name+' gives up moving aside');b.asideHold=null;
        if(b.resumeJob){b.state='go';b.path=mkPath([b.p].concat(b.resumeJob.path.pts));b.s=0}else{b.state=b.prevState==='done'?'done':'idle';b.path=null}b.knots=null}}
    for(const b of B){if(b.state!=='go'||b.mustYield==null||W.t-(b.lastAside||-9)<3)continue;const o=B[b.mustYield];if(!o||o===b)continue;
      if(o.state==='aside'&&o.asideFor===b.i)continue;
      if(prio(W,b)>prio(W,o)){aside(W,b,o);b.lastAside=W.t}}
    for(const b of B){if(b.state!=='go'||b.brakeSince==null||W.t-b.brakeSince<2||W.t-(b.lastAside||-9)<3)continue;const o=B[b.brakeBy];if(!o)continue;
      if(o.state==='aside'&&o.asideFor===b.i)continue;
      if(o.state==='idle'||o.state==='done'){aside(W,o,b);b.brakeSince=null;continue}
      if(prio(W,b)>prio(W,o)||(o.brakeSince!=null&&o.brakeBy===b.i&&prio(W,b)>=prio(W,o))){aside(W,b,o);b.lastAside=W.t;b.brakeSince=null}}
    for(const b of B){if(b.state!=='go'||b.waitSince===null||W.t-b.waitSince<STARVE||b.blockedBy===null)continue;
      const o=B[b.blockedBy];if(!o)continue;
      if(o.state==='aside'&&o.asideFor===b.i)continue;
      if(o.state==='idle'||o.state==='done'){aside(W,o,b);b.waitSince=W.t}
      else if(o.state==='go'&&o.waitSince!==null&&o.blockedBy===b.i){const lo=prio(W,b)>prio(W,o)?b:o;aside(W,lo,lo===b?o:b);b.waitSince=W.t;o.waitSince=W.t}
      else if(W.t-b.waitSince>3*STARVE){aside(W,b,o);b.waitSince=W.t}}
    B.forEach(swapOut)}
  for(const b of B){
    swapIn(b,obs[b.i]);
    let drive=false,want=b.f,speed=0,spinTo=null;b.holding=false;
    const ctl=()=>{
    if(moving(b)&&b.path){
      const[s,off]=project(b.path,b.p,b.s);b.s=Math.max(b.s,s);
      if(b.state==='go'&&b.job.kind==='orbit'&&W.t>b.job.until){finish(W,b);return}
      if(b.s>=b.path.total-3&&len(sub(b.p,b.path.pts[b.path.pts.length-1]))<6){
        if(b.state==='aside'){if(b.resumeJob){b.state='go';b.path=b.resumeJob.path;b.s=0}else{b.state=b.prevState==='done'?'done':'idle';b.path=null}b.knots=null;return}
        finish(W,b);return}
      const lim=b.knots?allowed(b,W.t+0.3):b.s+2;
      if(lim>b.s+1||!b.knots){
        const tgt=at(b.path,b.s+LOOK),end=b.path.pts[b.path.pts.length-1];const aim=(b.s>b.path.total-LOOK&&len(sub(end,b.p))<LOOK)?end:tgt;
        const d=sub(aim,b.p);want=brg(d[0],d[1]);const err=wrap(want-b.f);
        if(Math.abs(err)>(b.spd<2?30:60)){if(b.spd<2)spinTo=want}
        else{drive=true;const c=Math.cos(err*Math.PI/180);speed=S*Math.max(0.25,c*c)}}
      else b.holding=true;
    }
    b.brake=false;
    if(drive){const vi=mulv(dirOf(want),speed);
      for(const o of B){if(o===b)continue;const oo=obs[o.i];const d=sub(oo.p,b.p),w=sub(oo.v,vi),ww=dot(w,w),cl=dot(d,w)<0;
        let hit=len(d)<16&&cl;if(!hit&&PRED&&cl){const ts=ww<1e-6?0:clamp(-dot(d,w)/ww,0,1);hit=len(add(d,mulv(w,ts)))<PRED}
        if(hit){drive=false;b.brake=true;W.stats.brake++;if(b.brakeSince==null||b.brakeBy!==o.i){b.brakeSince=W.t;b.brakeBy=o.i}break}}}
    if(!b.brake&&drive)b.brakeSince=null;
    if(b.lastMoveObs===undefined){b.lastMoveObs=b.p.slice()}
    };ctl();
    swapOut(b);
    (b.cmdQ=b.cmdQ||[]).push({t:W.t+(REAL?CMDLAT:0),drive,want,speed,spinTo});
  }
  for(const b of B){
    let c=null;while(b.cmdQ&&b.cmdQ.length&&b.cmdQ[0].t<=W.t)c=b.cmdQ.shift();if(c)b.cmd=c;const cmd=b.cmd||{drive:false,want:b.f,speed:0,spinTo:null};
    if(cmd.spinTo!=null&&b.spd<2){
      if(!b.spinning){b.spinning=true;b.spinRate=REAL?SPINLO+(SPINHI-SPINLO)*W.rng():TURN;b.spinStall=REAL&&W.rng()<0.3?W.t+0.4:0}
      const err=wrap(cmd.spinTo-b.f);
      if(W.t>b.spinStall)b.f=(b.f+clamp(err,-b.spinRate*DT,b.spinRate*DT)+360)%360;
      b.spinDir=Math.sign(err)||1;
    }else{
      if(b.spinning){b.spinning=false;b.runon=REAL?b.spinDir*RUNON*W.rng():0}
      if(b.runon){const r=clamp(b.runon,-60*DT,60*DT);b.f=(b.f+r+360)%360;b.runon-=r;if(Math.abs(b.runon)<0.1)b.runon=0}
      if(cmd.drive){const e=wrap(cmd.want+(REAL?b.bias:0)-b.f)*(REAL?GAIN:1);b.f=(b.f+clamp(e,-SLEW*DT,SLEW*DT)+360)%360}
    }
    const target=cmd.drive?cmd.speed*(REAL?b.spdF:1):0;
    b.spd+=(target-b.spd)*Math.min(1,DT/(target>b.spd?TAU_UP:TAU_DN));if(b.spd<0.05)b.spd=0;
    b.v=mulv(dirOf(b.f),b.spd);b.p=add(b.p,mulv(b.v,DT));b.p=[clamp(b.p[0],BR,AW-BR),clamp(b.p[1],BR,AH-BR)];
    if(len(sub(b.p,b.lastMoveP))>8){b.lastMoveP=b.p.slice();b.lastMoveT=W.t;b.stuckFlag=false}
    if(b.state==='go'&&W.t-b.lastMoveT>20&&!b.stuckFlag){b.stuckFlag=true;W.stats.stuck++;log(W,b.name+' no progress 20s ('+b.job.kind+')')}
  }
  for(let i=0;i<B.length;i++)for(let j=i+1;j<B.length;j++){const d=len(sub(B[i].p,B[j].p));W.stats.minD=Math.min(W.stats.minD,d);
    const c=d<2*BR+0.5,nr=d<12;if(c&&!B[i].wasContact[j]){W.stats.contact++;log(W,'CONTACT '+B[i].name+'-'+B[j].name)}
    if(nr&&!B[i].wasContact['n'+j])W.stats.near++;B[i].wasContact[j]=c;B[i].wasContact['n'+j]=nr}
}
if(typeof module!=='undefined')module.exports={World,step,CASES};
