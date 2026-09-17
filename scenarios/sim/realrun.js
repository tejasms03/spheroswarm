const Q=require('./ptr_real.js');
const label=process.argv[2];
const out={label,cases:{},stress:{}};
for(const sc of Object.keys(Q.CASES)){const agg={done:0,started:0,contact:0,minD:1e9,asides:0,time:0,brake:0,frames:0};
  for(let seed=1;seed<=5;seed++){const W=Q.World(0,seed,sc);while(W.t<120&&!(W.stats.jobsDone===W.stats.jobsStarted&&W.balls.every(b=>b.spd<0.1)))Q.step(W);
    agg.done+=W.stats.jobsDone;agg.started+=W.stats.jobsStarted;agg.contact+=W.stats.contact;agg.minD=Math.min(agg.minD,W.stats.minD);agg.asides+=W.stats.asides;agg.time+=W.t;agg.brake+=W.stats.brake;agg.frames+=W.stats.frames}
  out.cases[sc]={finished:`${agg.done}/${agg.started}`,contacts:agg.contact,closest_cm:+agg.minD.toFixed(1),avg_time_s:+(agg.time/5).toFixed(1),asides_per_run:+(agg.asides/5).toFixed(1),brake_pct:+(100*agg.brake/agg.frames).toFixed(1)}}
for(const n of [2,3,4,5]){const a={done:0,started:0,contact:0,near:0,minD:1e9,brake:0,frames:0,stuck:0};const seeds=n<=3?12:6;
  for(let s=1;s<=seeds;s++){const W=Q.World(n,s,'stress');while(W.t<300)Q.step(W);const st=W.stats;a.done+=st.jobsDone;a.started+=st.jobsStarted;a.contact+=st.contact;a.near+=st.near;a.minD=Math.min(a.minD,st.minD);a.brake+=st.brake;a.frames+=st.frames;a.stuck+=st.stuck}
  out.stress[n]={jobs_done_pct:+(100*a.done/Math.max(1,a.started)).toFixed(0),contacts:a.contact,near_under_12cm:a.near,closest_cm:+a.minD.toFixed(1),brake_pct:+(100*a.brake/a.frames).toFixed(1),no_progress_20s:a.stuck,seeds}}
console.log(JSON.stringify(out));
