const fs=require('fs');const label=process.argv[2];
delete require.cache[require.resolve('./ptr_real.js')];const Q=require('./ptr_real.js');
const scen=JSON.parse(fs.readFileSync('scenarios_raw.json'));const out={};
for(const sc of scen){const a={done:0,started:0,contact:0,near:0,minD:1e9,asides:0,brake:0,frames:0,stuck:0,t:0};const N=5;
  for(let seed=1;seed<=N;seed++){const W=Q.World(0,seed,sc);
    while(W.t<sc.duration_s){Q.step(W);if(W.balls.every(b=>(!b.queue||b.queue.length===0)&&(b.state==='idle'||b.state==='done')&&b.job.until===Infinity))break}
    const st=W.stats;a.done+=st.jobsDone;a.started+=st.jobsStarted;a.contact+=st.contact;a.near+=st.near;a.minD=Math.min(a.minD,st.minD);a.asides+=st.asides;a.brake+=st.brake;a.frames+=st.frames;a.stuck+=st.stuck;a.t+=W.t}
  out[sc.file]={jobs_finished:`${a.done}/${a.started} over ${N} runs`,contacts:a.contact,passes_under_12cm:a.near,closest_cm:+a.minD.toFixed(1),move_asides_per_run:+(a.asides/N).toFixed(1),brake_pct_of_time:+(100*a.brake/Math.max(1,a.frames)).toFixed(1),no_progress_20s:a.stuck,avg_run_s:+(a.t/N).toFixed(0)}}
console.log(JSON.stringify({label,out}));
